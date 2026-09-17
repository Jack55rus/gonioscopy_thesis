import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TrabecularMeshworkDataset
from metrics import (
    update_confusion_matrix,
    metrics_from_confusion_matrix,
)
from models.factory import create_model
from postprocess import postprocess_prediction
from utils.common import count_parameters, save_json


def print_metrics(title, metrics):
    print(f"\n--- {title} ---")

    print(
        "Primary PTM/NPTM Dice:",
        metrics["primary_mean_dice"],
    )

    print(
        "All-class mean Dice:",
        metrics["all_classes_mean_dice"],
    )

    print(
        "PTM Dice:",
        metrics["ptm"]["dice"],
    )

    print(
        "NPTM Dice:",
        metrics["nptm"]["dice"],
    )

    print(
        "Merged TM Dice:",
        metrics["merged_tm_dice"],
    )


def main():
    dataset = TrabecularMeshworkDataset(
        split=config.TEST_SPLIT,
        training=False,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
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

    raw_confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.long,
    )

    postprocessed_confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.long,
    )

    primary_class_ids = [
        config.CLASS_NAMES.index(class_name)
        for class_name in config.PRIMARY_CLASS_NAMES
    ]

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="test"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            # -------------------------
            # Raw metrics
            # -------------------------
            update_confusion_matrix(
                raw_confusion_matrix,
                predictions.cpu(),
                targets.cpu(),
            )

            # -------------------------
            # Post-processing
            # -------------------------
            predictions_np = predictions.cpu().numpy()

            processed_predictions = []

            for prediction in predictions_np:
                processed = postprocess_prediction(
                    prediction=prediction,
                    class_ids=primary_class_ids,
                    min_size=config.POSTPROCESS_MIN_OBJECT_SIZE,
                    max_distance=config.POSTPROCESS_MAX_DISTANCE,
                )

                processed_predictions.append(
                    torch.from_numpy(processed)
                )

            processed_predictions = torch.stack(
                processed_predictions
            )

            update_confusion_matrix(
                postprocessed_confusion_matrix,
                processed_predictions,
                targets.cpu(),
            )

    raw_metrics = metrics_from_confusion_matrix(
        raw_confusion_matrix
    )

    postprocessed_metrics = metrics_from_confusion_matrix(
        postprocessed_confusion_matrix
    )

    common_info = {
        "model_name": config.MODEL_NAME,
        "parameters": count_parameters(model),
    }

    raw_metrics.update(common_info)

    postprocessed_metrics.update(
        {
            **common_info,
            "postprocess_min_object_size": (
                config.POSTPROCESS_MIN_OBJECT_SIZE
            ),
            "postprocess_max_distance": (
                config.POSTPROCESS_MAX_DISTANCE
            ),
        }
    )

    results = {
        "raw": raw_metrics,
        "postprocessed": postprocessed_metrics,
    }

    save_json(
        results,
        config.METRICS_PATH,
    )

    print_metrics(
        "RAW",
        raw_metrics,
    )

    print_metrics(
        "POST-PROCESSED",
        postprocessed_metrics,
    )

    print(
        "\nPost-processing:"
        f" min_size={config.POSTPROCESS_MIN_OBJECT_SIZE}px,"
        f" max_distance={config.POSTPROCESS_MAX_DISTANCE}px"
    )

    print(f"\nSaved: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
