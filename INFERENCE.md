# TM segmentation: run the final model on your images

`predict.py` runs the final pipeline of the thesis (FlexibleUNet-B1 + B4 ensemble, flip TTA with three
views, threshold 0.5, removal of components < 500 px) on any image or folder of GS-1 gonio-photographs.

## Setup

```bash
pip install torch monai numpy Pillow scikit-image
```

Download the two weight files into one folder (e.g. `weights/`):
`monai_flexible_unet_b1_best.pt` (35 MB) and `monai_flexible_unet_b4_best.pt` (85 MB).
Download link: https://drive.google.com/drive/folders/1CaIWB5j4dTlcAy3KeMngRbbJRhXthmZI?usp=sharing
SHA-256: b1 `e295fe71b970253407f2d9a29fbe0377a7908c7473fe7218362ccc7931e36929`, b4 `7adb897c0250e7b0fb4fbd8323673a30b139593408145be010ffc3bb75417e9c`

## Run

```bash
python predict.py --weights-dir weights/ image.jpg other_image.jpg folder_of_images/ --out results/
```

Options: `--device cpu|cuda` (default: GPU if available), `--save-prob` (also save the TM probability map),
`--force` (overwrite existing outputs).

For each image the script writes `<name>_mask.png` (TM = 255, same size as the input) and
`<name>_overlay.png` (TM in red over the image).

## Notes

- The networks were trained on rotated images (iris at the bottom, cornea at the top), as in the
  pre-processed version of the dataset; other orientations are not supported.
- Tested with Python 3.13, torch 2.11, MONAI 1.6, scikit-image 0.26, Pillow 12.3, numpy 2.5.
  On the 31 test images it reproduces the thesis results exactly (pooled TM Dice 0.835).
  About 0.12 s per image on an RTX 2080 Ti and a few seconds per image on CPU.
