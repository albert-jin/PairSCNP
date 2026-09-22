"""Train the single-view SCNP baseline or paired-view PairSCNP."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairscnp.paths import ROOT, read_json, write_json, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["topomortar", "crack500"], required=True)
    parser.add_argument("--method", choices=["scnp", "pairscnp"], default="pairscnp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/DeepLab_R-103.pkl")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.accum, args.val_every) < 1 or args.workers < 0:
        parser.error("Steps, batch, accumulation and validation interval must be positive.")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from pairscnp.model import PairSCNP
    from pairscnp.data import TrainingData, seed_worker, evaluate
    from pairscnp.chroma_consistency import paired_view, structure_js
    from pairscnp.amodal_completion import amodal_view
    from pairscnp.runtime import configure_runtime
    configure_runtime(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The recorded training recipe requires a BF16-capable CUDA GPU.")
    spec = read_json(ROOT / "configs/backbone.json")
    if sha256(args.weights) != spec["sha256"]:
        raise ValueError("Backbone checkpoint does not match the recorded ImageNet initialization.")
    manifest_path = ROOT / "data" / args.dataset / "prepared_manifest.json"
    manifest = read_json(manifest_path)
    for split in ("train", "val"):
        for row in manifest["splits"][split]:
            if sha256(ROOT / row["cache"]) != row["cache_sha256"]:
                raise ValueError(f"Cache changed: {row['id']}")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(learning_rate=0.001, weight_decay=0.1, scheduler="cosine", effective_batch=args.batch*args.accum)
    write_json(args.output / "config.json", config)
    model = PairSCNP(args.weights).cuda()
    write_json(args.output / "initialization.json", model.load_audit)
    (args.output / "model_config.yaml").write_text(model.config_text)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    generator = torch.Generator().manual_seed(args.seed + 12345)
    loader = DataLoader(TrainingData(manifest["splits"]["train"]), batch_size=args.batch,
        shuffle=True, drop_last=True, num_workers=args.workers, persistent_workers=args.workers>0,
        worker_init_fn=seed_worker, generator=generator, pin_memory=True)
    if not len(loader):
        raise ValueError("Training split is smaller than one batch.")
    iterator = iter(loader)
    views = torch.Generator(device="cuda").manual_seed(args.seed + 67890)
    occlusions = torch.Generator(device="cuda").manual_seed(args.seed + 9876)
    best = -1.0
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for folder in ("pairscnp", "scripts", "configs")
        for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".yaml", ".json")}
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = []
        for _ in range(args.accum):
            try:
                x, y, geometry, names = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, y, geometry, names = next(iterator)
            x, y, geometry = [v.cuda(non_blocking=True) for v in (x, y, geometry)]
            virtual_step = 3000 * step / args.steps
            if args.method == "scnp":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(x).float()
                supervised = model.criterion(z, y)
                js = z.new_zeros(())
                coefficient = 0.0
            else:
                x2 = paired_view(x, views)
                x2, _ = amodal_view(x2, y[:, None], geometry, occlusions, virtual_step)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z1, z2 = model(torch.cat((x, x2), dim=0)).float().chunk(2, dim=0)
                supervised = 0.5 * (model.criterion(z1, y) + model.criterion(z2, y))
                js = structure_js(z1, z2, y[:, None], geometry)
                coefficient = 0.5 * min(virtual_step / 600, 1.0)
            loss = supervised + coefficient * js
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / args.accum).backward()
            values.append([float(loss.detach()), float(supervised.detach()), float(js.detach())])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            record = dict(zip(("loss", "supervised", "js"), np.mean(values, axis=0).tolist()))
            record.update(step=step, js_weight=coefficient, lr=optimizer.param_groups[0]["lr"])
            with (args.output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.val_every == 0 or step == args.steps:
            validation = evaluate(model, manifest["splits"]["val"])
            validation["step"] = step
            with (args.output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(validation) + "\n")
            checkpoint = {"model":model.state_dict(), "step":step, "config":config,
                "optimizer":optimizer.state_dict(), "scheduler":scheduler.state_dict()}
            torch.save(checkpoint, args.output / "last.pt")
            if validation["mean"]["dice"] > best:
                best = validation["mean"]["dice"]
                torch.save(checkpoint, args.output / "best.pt")
                write_json(args.output / "best_validation.json", validation)
            print(json.dumps({"step":step, "validation_dice":validation["mean"]["dice"]}), flush=True)
    write_json(args.output / "evaluation_lock.json", {"checkpoint_sha256":sha256(args.output / "best.pt"),
        "manifest_sha256":sha256(manifest_path), "source_hashes":source_hashes,
        "selected_step":read_json(args.output / "best_validation.json")["step"],
        "selection":"maximum validation mean Dice"})

if __name__ == "__main__":
    main()
