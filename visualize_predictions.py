from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

import config
from dataset import TrabecularMeshworkDataset
from models.factory import create_model
from postprocess import postprocess_prediction


OUTPUT_DIR = config.OUTPUT_DIR / "prediction_visualizations" / f'{config.MODEL_NAME}'
NUMBER_OF_SAMPLES = 20

CLASS_NAMES = {
    0: "Background",
    1: "PTM",
    2: "NPTM",
}


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    image = image.cpu() * std + mean
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image.copy()

    # PTM: red channel
    ptm = mask == 1
    overlay[ptm, 0] = 1.0
    overlay[ptm, 1] *= 0.35
    overlay[ptm, 2] *= 0.35

    # NPTM: green channel
    nptm = mask == 2
    overlay[nptm, 0] *= 0.35
    overlay[nptm, 1] = 1.0
    overlay[nptm, 2] *= 0.35

    foreground = ptm | nptm
    result = image.copy()
    result[foreground] = (
        (1 - alpha) * image[foreground]
        + alpha * overlay[foreground]
    )

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = TrabecularMeshworkDataset(
        split=config.TEST_SPLIT,
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model().to(device)

    checkpoint = torch.load(
        config.CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    saved = 0

    primary_class_ids = [
        config.CLASS_NAMES.index(class_name)
        for class_name in config.PRIMARY_CLASS_NAMES
    ]

    with torch.inference_mode():
        for batch in loader:
            image_tensor = batch["image"][0]
            target = batch["target"][0].numpy()
            name = batch["name"][0]

            logits = model(
                batch["image"].to(device)
            )

            prediction = (
                logits.argmax(dim=1)[0]
                .cpu()
                .numpy()
            )

            prediction = postprocess_prediction(
                prediction=prediction,
                class_ids=primary_class_ids,
                min_size=config.POSTPROCESS_MIN_OBJECT_SIZE,
                max_distance=config.POSTPROCESS_MAX_DISTANCE,
            )

            image = denormalize(image_tensor)

            target_display = target.copy()
            target_display[
                target_display == config.IGNORE_INDEX
            ] = 0

            gt_overlay = create_overlay(
                image,
                target_display,
            )

            prediction_overlay = create_overlay(
                image,
                prediction,
            )

            figure, axes = plt.subplots(
                1,
                4,
                figsize=(18, 5),
            )

            axes[0].imshow(image)
            axes[0].set_title("Original")

            axes[1].imshow(target_display, vmin=0, vmax=2)
            axes[1].set_title("Ground truth")

            axes[2].imshow(gt_overlay)
            axes[2].set_title("Ground-truth overlay")

            axes[3].imshow(prediction_overlay)
            axes[3].set_title("Prediction overlay")

            for axis in axes:
                axis.axis("off")

            figure.suptitle(
                f"{name} | Red = PTM, Green = NPTM"
            )

            output_path = OUTPUT_DIR / f"{name}.png"

            plt.tight_layout()
            plt.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

            print(f"Saved: {output_path}")

            saved += 1

            if saved >= NUMBER_OF_SAMPLES:
                break


if __name__ == "__main__":
    main()
