from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TrabecularMeshworkDataset
from losses import SegmentationLoss
from metrics import (
    update_confusion_matrix,
    metrics_from_confusion_matrix,
)
from models.factory import create_model
from utils.common import count_parameters, save_json, set_seed


def run_validation(model, data_loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    confusion_matrix = torch.zeros(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
        dtype=torch.long,
    )

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="validation", leave=False):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            logits = model(images)
            loss = loss_function(logits, targets)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            update_confusion_matrix(
                confusion_matrix,
                predictions.cpu(),
                targets.cpu(),
            )

    metrics = metrics_from_confusion_matrix(confusion_matrix)
    metrics["loss"] = total_loss / len(data_loader.dataset)

    return metrics


def main():
    set_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_dataset = TrabecularMeshworkDataset(
        split=config.TRAIN_SPLIT,
        training=True,
    )

    validation_dataset = TrabecularMeshworkDataset(
        split=config.VAL_SPLIT,
        training=False,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model().to(device)
    loss_function = SegmentationLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_validation_dice = -1.0
    epochs_without_improvement = 0
    history = []

    print(f"Device: {device}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_training_loss = 0.0

        progress_bar = tqdm(
            training_loader,
            desc=f"epoch {epoch}/{config.EPOCHS}",
        )

        for batch in progress_bar:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = loss_function(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_training_loss += loss.item() * images.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        training_loss = (
            total_training_loss / len(training_dataset)
        )

        validation_metrics = run_validation(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        # validation_dice = (
        #     validation_metrics["macro_foreground_dice"]
        # )

        validation_dice = (
            validation_metrics["primary_mean_dice"]
        )

        scheduler.step(validation_dice)

        epoch_result = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_metrics["loss"],
            # "validation_macro_dice": validation_dice,
            "validation_primary_dice": validation_dice,
            "validation_all_classes_dice": validation_metrics[
                "all_classes_mean_dice"
            ],
            "validation_ptm_dice": validation_metrics["ptm"]["dice"],
            "validation_nptm_dice": validation_metrics["nptm"]["dice"],
            "validation_merged_tm_dice": validation_metrics[
                "merged_tm_dice"
            ],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(epoch_result)
        save_json(
            {"history": history},
            config.OUTPUT_DIR / f"{config.MODEL_NAME}_history.json",
        )

        print(
            f"epoch={epoch} "
            f"train_loss={training_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_dice={validation_dice:.4f} "
            f"ptm={validation_metrics['ptm']['dice']:.4f} "
            f"nptm={validation_metrics['nptm']['dice']:.4f}"
        )

        if validation_dice > best_validation_dice:
            best_validation_dice = validation_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": config.MODEL_NAME,
                    "parameters": count_parameters(model),
                    "validation_metrics": validation_metrics,
                },
                config.CHECKPOINT_PATH,
            )

            print(f"Saved: {config.CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= config.EARLY_STOPPING_PATIENCE
        ):
            print("Early stopping.")
            break


if __name__ == "__main__":
    main()
