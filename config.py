from pathlib import Path

# =========================
# DATA
# =========================
DATA_ROOT = Path(r"/home/mark/Downloads/thesis_dataset")

ORIGINALS_DIR = DATA_ROOT / "Originals_rotated"
ANNOTATIONS_DIR = DATA_ROOT / "Annotations_rotated"

# PTM_FOLDER = "Pigmented Trabecular Meshwork"
# NPTM_FOLDER = "Non-pigmented Trabecular Meshwork"
# PTM_SUFFIX = "_PTM"
# NPTM_SUFFIX = "_NPTM"
# # 0 = background, 1 = PTM, 2 = NPTM
# NUM_CLASSES = 3
IGNORE_INDEX = 255

##
# new part
USE_ONLY_PRIMARY_CLASSES = True
# PRIMARY_CLASS_DEFINITIONS = [
#     {
#         "name": "ptm",
#         "folder": "Pigmented Trabecular Meshwork",
#         "suffix": "_PTM",
#     },
#     {
#         "name": "nptm",
#         "folder": "Non-pigmented Trabecular Meshwork",
#         "suffix": "_NPTM",
#     },
# ]
#
# ALL_CLASS_DEFINITIONS = [
#     *PRIMARY_CLASS_DEFINITIONS,
#     {
#         "name": "cornea",
#         "folder": "Cornea",
#         "suffix": "_C",
#     },
#     {
#         "name": "scleral_spur",
#         "folder": "Scleral Spur",
#         "suffix": "_SS",
#     },
#     {
#         "name": "iris_root",
#         "folder": "Iris Root",
#         "suffix": "_IR",
#     },
#     {
#         "name": "ciliary_body_band",
#         "folder": "Ciliary Body Band",
#         "suffix": "_CBB",
#     },
#     {
#         "name": "synechia",
#         "folder": "Synechia",
#         "suffix": "_SY",
#     },
# ]
#
# CLASS_DEFINITIONS = (
#     PRIMARY_CLASS_DEFINITIONS
#     if USE_ONLY_PRIMARY_CLASSES
#     else ALL_CLASS_DEFINITIONS
# )
#
# CLASS_NAMES = [
#     "background",
#     *[item["name"] for item in CLASS_DEFINITIONS],
# ]
#
# NUM_CLASSES = len(CLASS_NAMES)
#
# PRIMARY_CLASS_NAMES = ["ptm", "nptm"]

CLASS_DEFINITIONS = [
    {
        "name": "tm",
        "sources": [
            {
                "folder": "Pigmented Trabecular Meshwork",
                "suffix": "_PTM",
            },
            {
                "folder": "Non-pigmented Trabecular Meshwork",
                "suffix": "_NPTM",
            },
        ],
    },
]

CLASS_NAMES = [
    "background",
    "tm",
]

NUM_CLASSES = 2

PRIMARY_CLASS_NAMES = ["tm"]

##

TRAIN_SPLIT = "Training"
VAL_SPLIT = "Validation"
TEST_SPLIT = "Test"


# Original images are 1280x960. This preserves the 4:3 ratio.
IMAGE_HEIGHT = 384 * 2
IMAGE_WIDTH = 512 * 2


# If PTM and NPTM overlap, ignore those pixels during training.
IGNORE_OVERLAPS = True

# Samples missing either requested mask are skipped.
SKIP_INCOMPLETE_SAMPLES = False

# =========================
# TRAINING
# =========================
# MODEL_NAME = "unet"
# depthwise_unet tiny_unet unet
# monai update
USE_PRETRAINED_WEIGHTS = True
# MODEL_NAME = "monai_unet"
# MODEL_NAME = "monai_basic_unet"
# MODEL_NAME = "monai_segresnet"
MODEL_NAME = "monai_flexible_unet_b0"
# MODEL_NAME = "monai_flexible_unet_b1"
# MODEL_NAME = "lraspp_mobilenet"
# MODEL_NAME = "fast_scnn"
#####


BASE_CHANNELS = 16 # 16

BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 100

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DICE_LOSS_WEIGHT = 0.7
CROSS_ENTROPY_WEIGHT = 0.3

USE_AMP = False
EARLY_STOPPING_PATIENCE = 25

# =========================
# AUGMENTATION
# =========================
HORIZONTAL_FLIP_PROBABILITY = 0.5
MAX_ROTATION_DEGREES = 8
BRIGHTNESS_JITTER = 0.15
CONTRAST_JITTER = 0.15

# =========================
# OUTPUTS
# =========================
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / f"{MODEL_NAME}_best.pt"
METRICS_PATH = OUTPUT_DIR / f"{MODEL_NAME}_test_metrics.json"
BENCHMARK_PATH = OUTPUT_DIR / f"{MODEL_NAME}_benchmark.json"

SEED = 42

POSTPROCESS_MIN_OBJECT_SIZE = 10_000

# After small-object removal, find the largest object of each class.
# Any other component farther than this minimum Euclidean distance
# from the largest object is removed.
POSTPROCESS_MAX_DISTANCE = 100
