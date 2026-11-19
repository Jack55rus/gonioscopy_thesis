import torch


def model_forward(model, batch, device):
    """
    One shared forward helper for train/evaluate/visualization.

    This is intentionally tiny so metadata support does not leak throughout
    the codebase.
    """
    images = batch["image"].to(device)

    metadata = batch.get("metadata")
    if metadata is not None:
        metadata = metadata.to(device)

    return model(images, metadata) if metadata is not None else model(images)
