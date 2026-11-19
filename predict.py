"""Segment the trabecular meshwork (TM) in NIDEK GS-1 gonio-photographs.

Runs the final pipeline of the thesis on any image or folder of images:
  - two MONAI FlexibleUNet networks (EfficientNet-B1 and EfficientNet-B4 encoders),
  - test-time augmentation with three views (original, horizontal flip, vertical flip),
  - softmax probabilities averaged over views and networks,
  - TM probability thresholded at 0.5,
  - connected components smaller than 500 pixels removed (8-connectivity),
  - mask resized back to the original image size.

The networks were trained on rotated images (iris at the bottom, cornea at the top), as in the
pre-processed version of the dataset, so input images should have the same orientation.

Usage:
    python predict.py --weights-dir weights/ image.jpg [more images or folders] --out results/

The weights directory must contain monai_flexible_unet_b1_best.pt and monai_flexible_unet_b4_best.pt.
For each input image the script writes <name>_mask.png (TM = 255) and <name>_overlay.png
(TM in red over the image); --save-prob also writes the averaged TM probability as <name>_prob.png.
Existing output files are never overwritten unless --force is given.

Requirements: torch, monai, numpy, Pillow, scikit-image.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import FlexibleUNet
from PIL import Image
from skimage.morphology import remove_small_objects

NUM_CLASSES = 7          # background, TM, cornea, scleral spur, iris root, ciliary body band, synechia
TM_CLASS = 1
INPUT_SIZE = (384, 512)  # height, width used for training
THRESHOLD = 0.5
MIN_SIZE = 500           # pixels, at the 384 x 512 resolution
TTA_VIEWS = ("none", "hflip", "vflip")
MEAN = (0.485, 0.456, 0.406)  # ImageNet normalisation, as in training
STD = (0.229, 0.224, 0.225)
NETWORKS = {
    "monai_flexible_unet_b1_best.pt": ("efficientnet-b1", (160, 80, 40, 20, 10)),
    "monai_flexible_unet_b4_best.pt": ("efficientnet-b4", (220, 120, 60, 30, 15)),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def build_network(backbone, decoder_channels):
    return FlexibleUNet(
        spatial_dims=2, in_channels=3, out_channels=NUM_CLASSES, backbone=backbone, pretrained=False,
        decoder_channels=decoder_channels, norm=("batch", {"eps": 1e-3, "momentum": 0.1}),
        act=("relu", {"inplace": True}), dropout=0.0, upsample="nontrainable", interp_mode="bilinear",
        is_pad=True,
    )


def load_networks(weights_dir, device):
    """Load the two networks. The checkpoints also contain the metadata refinement head, which the
    final pipeline does not use (its last layer is all zero), so only the base network is kept."""
    networks = []
    for filename, (backbone, decoder) in NETWORKS.items():
        path = Path(weights_dir) / filename
        if not path.exists():
            sys.exit(f"missing weights: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model_state", state)
        head = [k for k in state if k.startswith("refiner.") and k.endswith(("8.weight", "8.bias"))]
        if any(state[k].abs().max() > 0 for k in head):
            warnings.warn(f"{filename}: the metadata head is not zero; it is ignored here")
        base = {k[len("base_model."):]: v for k, v in state.items() if k.startswith("base_model.")} or state
        network = build_network(backbone, decoder)
        network.load_state_dict(base)
        networks.append(network.to(device).eval())
    return networks


def preprocess(image):
    resized = image.resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    return x.unsqueeze(0)


def flip(x, view):
    if view == "hflip":
        return x.flip(3)
    if view == "vflip":
        return x.flip(2)
    return x


@torch.inference_mode()
def tm_probability(networks, x):
    """Averaged TM probability (H x W, at the 384 x 512 input resolution)."""
    probs = []
    for network in networks:
        for view in TTA_VIEWS:
            logits = network(flip(x, view))
            probs.append(torch.softmax(flip(logits, view), dim=1)[0, TM_CLASS])
    return torch.stack(probs).mean(0).cpu().numpy()


def postprocess(prob):
    mask = prob > THRESHOLD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # newer scikit-image renames min_size; the behaviour is the same
        return remove_small_objects(mask, min_size=MIN_SIZE, connectivity=2)


def segment(networks, image, device):
    """Return (mask, prob) at the 384 x 512 resolution for a PIL RGB image."""
    prob = tm_probability(networks, preprocess(image).to(device))
    return postprocess(prob), prob


def collect_inputs(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            files.append(p)
        else:
            print(f"skipping {p}", file=sys.stderr)
    return files


def save(array_image, path, force):
    if path.exists() and not force:
        print(f"exists, not overwritten: {path}", file=sys.stderr)
        return
    array_image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--out", default="tm_predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-prob", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        sys.exit("no input images found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    networks = load_networks(args.weights_dir, device)

    for f in files:
        image = Image.open(f).convert("RGB")
        mask, prob = segment(networks, image, device)
        full_mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
        save(full_mask, out / f"{f.stem}_mask.png", args.force)
        overlay = np.asarray(image).copy()
        sel = np.asarray(full_mask) > 0
        overlay[sel] = (0.45 * overlay[sel] + 0.55 * np.array([230, 30, 30])).astype(np.uint8)
        save(Image.fromarray(overlay), out / f"{f.stem}_overlay.png", args.force)
        if args.save_prob:
            full_prob = Image.fromarray((prob * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
            save(full_prob, out / f"{f.stem}_prob.png", args.force)
        print(f"{f.name}: TM {'found' if mask.any() else 'not found'} ({int(np.asarray(full_mask).astype(bool).sum())} px)")


if __name__ == "__main__":
    main()
