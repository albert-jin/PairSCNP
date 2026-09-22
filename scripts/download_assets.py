"""Download the pinned TopoMortar files or the public backbone checkpoint."""
from pathlib import Path
import argparse
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pairscnp.paths import ROOT, read_json, sha256

def fetch(url, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if sha256(temporary) != expected:
        raise ValueError(f"Checksum mismatch: {destination.name}")
    temporary.replace(destination)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=["topomortar", "backbone"])
    args = parser.parse_args()
    if args.asset == "backbone":
        spec = read_json(ROOT / "configs/backbone.json")
        fetch(spec["url"], ROOT / "weights/DeepLab_R-103.pkl", spec["sha256"])
    else:
        for name in ("train_val_download.json", "test_download.json"):
            for record in read_json(ROOT / "data/topomortar" / name)["files"]:
                path = Path(record["path"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Invalid download path")
                fetch(record["url"], ROOT / "data/topomortar/raw" / path, record["sha256"])
    print("Downloads and SHA256 checks complete.")

if __name__ == "__main__":
    main()
