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
from skimage.morphology import remove_small_objects, binary_erosion, disk


TTA_MODES = [
    "none",
    # "hflip",
    # "vflip",
    "hvflip",
]

ENSEMBLE_MODELS = [
    {
        "name": "monai_flexible_unet_b1",
        "checkpoint": config.OUTPUT_DIR / "monai_flexible_unet_b1_best.pt",
    },
    {
        "name": "monai_flexible_unet_b4",
        "checkpoint": config.OUTPUT_DIR / "monai_flexible_unet_b4_best.pt",
    },
    # {
    #     "name": "monai_flexible_unet_b7",
    #     "checkpoint": config.OUTPUT_DIR / "monai_flexible_unet_b7_best.pt",
    # },
    # {
    #     "name": "lraspp_mobilenet",
    #     "checkpoint": config.OUTPUT_DIR / "lraspp_mobilenet_best.pt",
    # },
]

OUTPUT_DIR = config.OUTPUT_DIR / "prediction_visualizations" / "ensemble_tta"
NUMBER_OF_SAMPLES = 40
TM_CLASS_ID = 1
TM_THRESHOLD = 0.5

def transform_image(image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(image, dims=[3])
    if mode == "vflip":
        return torch.flip(image, dims=[2])
    if mode == "hvflip":
        return torch.flip(image, dims=[2, 3])
    return image

def undo_transform(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(logits, dims=[3])
    if mode == "vflip":
        return torch.flip(logits, dims=[2])
    if mode == "hvflip":
        return torch.flip(logits, dims=[2, 3])
    return logits

def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image[:3]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = image.cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()

def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    tm = mask == TM_CLASS_ID
    overlay = image.copy()
    overlay[tm, 0] = 1.0
    overlay[tm, 1] *= 0.35
    overlay[tm, 2] *= 0.35
    result = image.copy()
    result[tm] = (1 - alpha) * image[tm] + alpha * overlay[tm]
    return result

def remove_small_tm_objects(prediction: np.ndarray, min_size: int = 100) -> np.ndarray:
    tm_mask = prediction == TM_CLASS_ID
    tm_mask = remove_small_objects(tm_mask, min_size=min_size, connectivity=2)
    result = prediction.copy()
    result[result == TM_CLASS_ID] = 0
    result[tm_mask] = TM_CLASS_ID
    return result

def calculate_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    valid = target != config.IGNORE_INDEX
    prediction_tm = (prediction == TM_CLASS_ID) & valid
    target_tm = (target == TM_CLASS_ID) & valid
    intersection = np.logical_and(prediction_tm, target_tm).sum()
    denominator = prediction_tm.sum() + target_tm.sum()
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator

def calculate_pixel_stats(prediction: np.ndarray, target: np.ndarray) -> dict:
    valid = target != config.IGNORE_INDEX
    prediction_tm = (prediction == TM_CLASS_ID) & valid
    target_tm = (target == TM_CLASS_ID) & valid
    tp = np.logical_and(prediction_tm, target_tm).sum()
    fp = np.logical_and(prediction_tm, ~target_tm & valid).sum()
    fn = np.logical_and(~prediction_tm & valid, target_tm).sum()
    tn = np.logical_and(~prediction_tm & valid, ~target_tm & valid).sum()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "iou": iou,
    }

def load_ensemble(device):
    models = []
    original_model_name = config.MODEL_NAME
    for model_config in ENSEMBLE_MODELS:
        config.MODEL_NAME = model_config["name"]
        model = create_model().to(device)
        checkpoint = torch.load(
            model_config["checkpoint"],
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
        print(f"Loaded: {model_config['name']} from {model_config['checkpoint']}")
    config.MODEL_NAME = original_model_name
    return models

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(device)
    saved = 0
    dices = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    correct_pred_dice_50 = 0
    correct_pred_iou_50 = 0
    with torch.inference_mode():
        for batch in loader:
            image_tensor = batch["image"][0]
            target = batch["target"][0].numpy()
            name = batch["name"][0]
            model_probabilities = []
            for model in models:
                tta_probabilities = []
                for tta_mode in TTA_MODES:
                    tta_batch = dict(batch)
                    tta_batch["image"] = transform_image(
                        batch["image"],
                        tta_mode,
                    )
                    logits = model_forward(model, tta_batch, device)
                    logits = undo_transform(logits, tta_mode)
                    tta_probabilities.append(torch.softmax(logits, dim=1))
                model_probability = torch.stack(
                    tta_probabilities
                ).mean(dim=0)
                model_probabilities.append(model_probability)
            probabilities = torch.stack(model_probabilities).mean(dim=0)

            tm_probability = probabilities[0, TM_CLASS_ID]
            confidence_map = tm_probability.cpu().numpy()
            prediction = (tm_probability > TM_THRESHOLD).long().cpu().numpy()
            prediction = remove_small_tm_objects(
                prediction,
                min_size=config.POSTPROCESS_MIN_OBJECT_SIZE,
            )
            # fdf
            # tm_mask = prediction == TM_CLASS_ID
            # tm_mask = binary_erosion(tm_mask, footprint=disk(3))
            # prediction = tm_mask.astype(np.uint8) * TM_CLASS_ID
            # ff
            dice = calculate_dice(prediction, target)
            if dice > 0.5:
                correct_pred_dice_50 += 1
            stats = calculate_pixel_stats(prediction, target)
            total_tp += stats["tp"]
            total_fp += stats["fp"]
            total_fn += stats["fn"]
            total_tn += stats["tn"]
            if stats['iou'] > 0.5:
                correct_pred_iou_50 += 1
            image = denormalize(image_tensor)
            target_display = (target == TM_CLASS_ID).astype(np.uint8)
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
                f"{name} | Dice={dice:.4f} | Precision={stats['precision']:.4f} | "
                f"Recall={stats['recall']:.4f} | IoU={stats['iou']:.4f}\n"
                f"TP={stats['tp']} | FP={stats['fp']} | FN={stats['fn']} | "
                f"TN={stats['tn']} | Threshold={TM_THRESHOLD}"
            )
            output_path = OUTPUT_DIR / f"{name}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(
                f"{name}: Dice={dice:.4f} "
                f"TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']} "
                f"Precision={stats['precision']:.4f} Recall={stats['recall']:.4f} "
                f"IoU={stats['iou']:.4f}"
            )
            dices.append(dice)
            saved += 1
            if saved >= NUMBER_OF_SAMPLES:
                break
    total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    total_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if 2 * total_tp + total_fp + total_fn > 0 else 1.0
    total_iou = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn > 0 else 1.0
    print(f"Mean per-image TM Dice: {np.mean(dices):.4f}")
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Total TN: {total_tn}")
    print(f"Global Precision: {total_precision:.4f}")
    print(f"Global Recall: {total_recall:.4f}")
    print(f"Global Dice: {total_dice:.4f}")
    print(f"Global IoU: {total_iou:.4f}")
    print(f'Accuracy = {(correct_pred_dice_50 / len(dices)):.4f}')
    print(f'Accuracy = {(correct_pred_iou_50 / len(dices)):.4f}')

if __name__ == "__main__":
    main()