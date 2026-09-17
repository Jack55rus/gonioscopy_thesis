# PTM / NPTM Segmentation Project

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
