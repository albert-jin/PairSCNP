"""Match downloaded images against hashes and create portable training caches."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairscnp.paths import ROOT, read_json, write_json, sha256

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["topomortar", "crack500"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from pairscnp.geometry import structure_prior
    manifest = read_json(ROOT / "data" / args.dataset / "manifest.json")
    index = {}
    for path in args.raw_root.resolve().rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path)
    for split, rows in manifest["splits"].items():
        for row in rows:
            for field in ("image", "mask"):
                candidates = index.get(Path(row[field]).name, [])
                matches = [p for p in candidates if sha256(p) == row[field + "_sha256"]]
                if not matches:
                    raise FileNotFoundError(f"No hash-matching {field} for {split}/{row['id']}")
                row[field] = str(matches[0])
            if split == "test":
                continue
            image = Image.open(row["image"]).convert("RGB")
            if row.get("image_rotation_ccw"):
                if split != "train" or row["image_rotation_ccw"] != 90:
                    raise ValueError(f"Unsupported orientation correction: {row['id']}")
                image = image.transpose(Image.Transpose.ROTATE_90)
            mask = np.asarray(Image.open(row["mask"]).convert("L")) > row["mask_threshold"]
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image/mask dimensions differ for {row['id']}")
            if args.dataset == "crack500":
                width, height = image.size
                scale = min(1., 1024 / max(width, height))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = image.resize(size, Image.Resampling.BILINEAR)
                mask = np.asarray(Image.fromarray(mask.astype("uint8")).resize(size, Image.Resampling.NEAREST)).copy()
            geometry = structure_prior(mask)
            cache = ROOT / row["cache"]
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, image=np.asarray(image), mask=mask.astype("uint8"), geometry=geometry.astype("float16"))
            row["cache_sha256"] = sha256(cache)
        print(f"Prepared {split}: {len(rows)} images", flush=True)
    dest = ROOT / "data" / args.dataset / "prepared_manifest.json"
    write_json(dest, manifest)
    print(dest)

if __name__ == "__main__":
    main()
