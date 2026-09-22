"""Run single-image inference from a PairSCNP training checkpoint."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from pairscnp.model import PairSCNP
    from pairscnp.data import predict
    from pairscnp.runtime import configure_runtime
    configure_runtime()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PairSCNP().cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    probability = predict(model, Image.open(args.image).convert("RGB"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((probability >= 0.5).astype("uint8") * 255).save(args.output)

if __name__ == "__main__":
    main()
