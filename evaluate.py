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
from utils.common import count_parameters, save_json


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

    confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.long,
    )

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="test"):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            update_confusion_matrix(
                confusion_matrix,
                predictions.cpu(),
                targets.cpu(),
            )

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["model_name"] = config.MODEL_NAME
    metrics["parameters"] = count_parameters(model)

    save_json(metrics, config.METRICS_PATH)

    # print(metrics)
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
    print(f"Saved: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
