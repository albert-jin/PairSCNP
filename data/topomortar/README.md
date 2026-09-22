# TopoMortar

Source: [official TopoMortar repository](https://github.com/jmlipman/TopoMortar), revision `b1f31f20a41b20aa775c21078b2c432c6dcc26b4`. Follow the upstream dataset license and citation instructions.

The supplied split contains **50 training, 20 validation and 350 test images**, as recorded in `manifest.json`. Training crops are sampled online; this manifest does not contain 4200 independent training images.

From the repository root:

```bash
python scripts/download_assets.py topomortar
python scripts/prepare_data.py --dataset topomortar --raw-root data/topomortar/raw
```

Alternatively download the upstream dataset at the pinned revision and pass its directory to `--raw-root`. Keep both `images` and `accurate` masks from all three splits. Download URLs and hashes are listed in `train_val_download.json` and `test_download.json`. This repository does not include the dataset binaries.
