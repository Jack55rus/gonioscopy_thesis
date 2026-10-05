from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import remove_small_objects
import config
from dataset import TrabecularMeshworkDataset
from model_forward import model_forward
from models.factory import create_model

OUTPUT_DIR = config.OUTPUT_DIR / "prediction_visualizations" / config.MODEL_NAME
NUMBER_OF_SAMPLES = 20
TM_CLASS_ID = 1

def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image[:3]
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
    tm = mask == TM_CLASS_ID
    overlay = image.copy()
    overlay[tm, 0] = 1.0
    overlay[tm, 1] *= 0.35
    overlay[tm, 2] *= 0.35
    result = image.copy()
    result[tm] = (1 - alpha) * image[tm] + alpha * overlay[tm]
    return result

def remove_small_tm_objects(
    prediction: np.ndarray,
    min_size: int = 100,
) -> np.ndarray:
    tm_mask = prediction == TM_CLASS_ID
    tm_mask = remove_small_objects(
        tm_mask,
        min_size=min_size,
        connectivity=2,
    )
    result = np.zeros_like(prediction)
    result[tm_mask] = TM_CLASS_ID
    return result

def calculate_dice(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    valid = target != config.IGNORE_INDEX
    prediction_tm = (prediction == TM_CLASS_ID) & valid
    target_tm = (target == TM_CLASS_ID) & valid
    intersection = np.logical_and(prediction_tm, target_tm).sum()
    denominator = prediction_tm.sum() + target_tm.sum()
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator

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
    dices = []
    with torch.inference_mode():
        for batch in loader:
            image_tensor = batch["image"][0]
            target = batch["target"][0].numpy()
            name = batch["name"][0]
            logits = model_forward(model, batch, device)
            probabilities = torch.softmax(logits, dim=1)
            confidence_map = probabilities[0, TM_CLASS_ID].cpu().numpy()
            prediction = logits.argmax(dim=1)[0].cpu().numpy()

            prediction = remove_small_tm_objects(
                prediction,
                min_size=config.POSTPROCESS_MIN_OBJECT_SIZE,
            )
            prediction = remove_small_tm_objects(
                prediction,
                min_size=config.POSTPROCESS_MIN_OBJECT_SIZE,
            )
            dice = calculate_dice(prediction, target)
            image = denormalize(image_tensor)
            target_display = target.copy()
            target_display[target_display == config.IGNORE_INDEX] = 0
            gt_overlay = create_overlay(image, target_display)
            prediction_overlay = create_overlay(image, prediction)

            figure, axes = plt.subplots(1, 5, figsize=(18, 5))
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[1].imshow(target_display, vmin=0, vmax=1, cmap="gray")
            axes[1].set_title("Ground truth")
            axes[2].imshow(gt_overlay)
            axes[2].set_title("Ground-truth overlay")
            axes[3].imshow(prediction_overlay)
            axes[3].set_title("Prediction overlay")

            confidence_plot = axes[4].imshow(
                confidence_map,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            axes[4].set_title("TM probability")
            figure.colorbar(
                confidence_plot,
                ax=axes[4],
                fraction=0.046,
                pad=0.04,
            )

            for axis in axes:
                axis.axis("off")

            figure.suptitle(
                f"{name} | TM Dice = {dice:.4f} | Red = TM"
            )
            output_path = OUTPUT_DIR / f"{name}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"{name}: Dice={dice:.4f} -> {output_path}")
            dices.append(dice)
            saved += 1

            if saved >= NUMBER_OF_SAMPLES:
                break
    print(np.mean(dices))

if __name__ == "__main__":
    main()