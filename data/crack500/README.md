# Crack500

Download the images and masks using the links in the [official pavement-crack-detection repository](https://github.com/fyangneil/pavement-crack-detection). A recorded alternative is [this Crack500 Kaggle mirror, version 1](https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001). Access and use are subject to the source's terms; data are not redistributed here.

Extract the archive locally, preserving the original image and mask filenames, and run from the repository root:

```bash
python scripts/prepare_data.py --dataset crack500 --raw-root /path/to/extracted/CRACK500
```

The supplied manifest selects **243 training, 49 validation and 200 test images** from the original 250/50/200 split, excluding the overlapping image families removed in the recorded experiments. Exact membership and image/mask hashes are in `manifest.json`. Extra files in the download are ignored. Grayscale masks use threshold 127. Training caches preserve aspect ratio and cap the longer side at 1024 pixels; training samples 512-pixel crops. Evaluation predicts at the original image resolution after sliding-window inference.

Three training photographs require a recorded 90-degree counterclockwise rotation to align with their masks. The preparation script applies the `image_rotation_ccw` entries after validating the original file hashes; masks and test images are not rotated.
