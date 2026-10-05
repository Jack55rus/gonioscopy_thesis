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
from postprocess import postprocess_prediction

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

            # primary_class_ids = [
            #     config.CLASS_NAMES.index(class_name)
            #     for class_name in config.PRIMARY_CLASS_NAMES
            # ]
            # predictions = predictions.cpu().numpy()
            # predictions = postprocess_prediction(
            #     prediction=predictions,
            #     class_ids=primary_class_ids,
            #     min_size=config.POSTPROCESS_MIN_OBJECT_SIZE,
            #     max_distance=config.POSTPROCESS_MAX_DISTANCE,
            # )
            predictions = torch.from_numpy(predictions)

            update_confusion_matrix(
                confusion_matrix,
                predictions.cpu(),
                targets.cpu(),
            )

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["model_name"] = config.MODEL_NAME
    metrics["parameters"] = count_parameters(model)

    save_json(metrics, config.METRICS_PATH)

    print(
        "Primary PTM/NPTM Dice:",
        metrics["primary_mean_dice"],
    )

    print(
        "All-class mean Dice:",
        metrics["all_classes_mean_dice"],
    )

    print(
        "TM Dice:",
        metrics["tm"]["dice"],
    )
    print(f"Saved: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
