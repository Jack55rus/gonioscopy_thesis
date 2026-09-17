import config

from models.tiny_unet import TinyUNet
from models.unet import UNet


def create_model():
    if config.MODEL_NAME == "tiny_unet":
        return TinyUNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
        )

    if config.MODEL_NAME == "depthwise_unet":
        return UNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
            depthwise=True,
        )

    if config.MODEL_NAME == "unet":
        return UNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
            depthwise=False,
        )

    raise ValueError(f"Unknown model: {config.MODEL_NAME}")
