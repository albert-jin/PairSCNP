"""Replay validation before evaluating a locked checkpoint on the test split."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairscnp.paths import ROOT, read_json, write_json, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reported-ids", action="store_true", help="Evaluate only the archived example evaluation IDs.")
    args = parser.parse_args()
    import torch
    from pairscnp.model import PairSCNP
    from pairscnp.data import evaluate
    from pairscnp.runtime import configure_runtime
    config = read_json(args.run / "config.json")
    configure_runtime(config["seed"])
    lock = read_json(args.run / "evaluation_lock.json")
    manifest_path = ROOT / "data" / config["dataset"] / "prepared_manifest.json"
    if sha256(args.run / "best.pt") != lock["checkpoint_sha256"] or sha256(manifest_path) != lock["manifest_sha256"]:
        raise ValueError("Checkpoint or data manifest changed after validation selection.")
    for name, expected in lock["source_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source changed after training: {name}")
    manifest = read_json(manifest_path)
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    model = PairSCNP().cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    replay = evaluate(model, manifest["splits"]["val"])
    expected = {r["id"]:r["dice"] for r in read_json(args.run / "best_validation.json")["per_image"]}
    error = max(abs(r["dice"] - expected[r["id"]]) for r in replay["per_image"])
    if error > 1e-4:
        raise ValueError(f"Validation replay differs by {error:.6g}")
    rows = manifest["splits"]["test"]
    if args.reported_ids:
        ids = read_json(ROOT / "results/reported_evaluation_ids.json")[config["dataset"]]
        by_id = {r["id"]:r for r in rows}
        rows = [by_id[i] for i in ids]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "validation_replay.json", {"max_dice_error":error})
    result = evaluate(model, rows, args.output)
    result["scope"] = "archived evaluation IDs" if args.reported_ids else "complete test split"
    result["checkpoint_sha256"] = lock["checkpoint_sha256"]
    write_json(args.output / "metrics.json", result)
    print(result["mean"])

if __name__ == "__main__":
    main()
