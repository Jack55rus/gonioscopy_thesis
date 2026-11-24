# PTM / NPTM Segmentation Project

## Thesis code, trained weights and inference

This repository contains the code of the Master's thesis *Automatic Segmentation of the Trabecular Meshwork
Using Deep Learning* (Evgenii Markin, University of Padova, 2026).

- Code: https://github.com/Jack55rus/gonioscopy_thesis
- Trained weights of the final ensemble (`monai_flexible_unet_b1_best.pt`, `monai_flexible_unet_b4_best.pt`):
  https://drive.google.com/drive/folders/1CaIWB5j4dTlcAy3KeMngRbbJRhXthmZI?usp=sharing
- To segment your own images with the final pipeline, see [INFERENCE.md](INFERENCE.md):
  `python predict.py --weights-dir weights/ image.jpg --out results/`

This is a normal Python project with no command-line arguments and no YAML.

Edit only `config.py`, then run the scripts directly.

## Dataset structure

```text
root/
├── originals_rotated/
│   ├── Training/
│   ├── Validation/
│   └── Test/
└── annotation_rotated/
    ├── Pigmented Trabecular Meshwork/
    │   ├── Training/
    │   ├── Validation/
    │   └── Test/
    └── Non-pigmented Trabecular Meshwork/
        ├── Training/
        ├── Validation/
        └── Test/
```

Filename example:

```text
DS01_04_GS-Face03-Shot11_OA_DI_HP.jpg
DS01_04_GS-Face03-Shot11_OA_DI_HP_PTM.png
DS01_04_GS-Face03-Shot11_OA_DI_HP_NPTM.png
```

## Classes

```text
0 = background
1 = pigmented trabecular meshwork
2 = non-pigmented trabecular meshwork
```

## Run

First edit:

```python
DATA_ROOT = Path(r"/absolute/path/to/your/root")
```

Then:

```bash
python inspect_dataset.py
python train.py
python evaluate.py
python benchmark.py
python compare_models.py
```

## Models

Choose one in `config.py`:

```python
MODEL_NAME = "tiny_unet"
MODEL_NAME = "depthwise_unet"
MODEL_NAME = "unet"
```

The depthwise U-Net is the main lightweight candidate.

## Metrics

The evaluator reports:

- Dice for PTM
- Dice for NPTM
- IoU
- precision
- recall
- specificity
- pixel accuracy
- macro foreground Dice
- merged TM Dice

Merged TM Dice treats PTM and NPTM as one structure. It is useful because it separates:

- failure to locate the trabecular meshwork
- confusion between its pigmented and non-pigmented parts
