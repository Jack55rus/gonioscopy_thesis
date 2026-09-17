import config


AVAILABLE_MODELS = [
    "tiny_unet",
    "depthwise_unet",
    "unet",
    "fast_scnn",
    "lraspp_mobilenet",
    "deeplab_mobilenet",
    "monai_unet",
    "monai_basic_unet",
    "monai_segresnet",
    "monai_flexible_unet_b0",
    "monai_flexible_unet_b1",
]


def create_model():
    name = config.MODEL_NAME

    if name == "tiny_unet":
        from models.tiny_unet import TinyUNet

        return TinyUNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
        )

    if name == "depthwise_unet":
        from models.unet import UNet

        return UNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
            depthwise=True,
        )

    if name == "unet":
        from models.unet import UNet

        return UNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
            depthwise=False,
        )

    if name == "fast_scnn":
        from models.fast_scnn import FastSCNN

        return FastSCNN(
            num_classes=config.NUM_CLASSES,
        )

    if name == "lraspp_mobilenet":
        from models.torchvision_segmentation import create_lraspp

        return create_lraspp(
            num_classes=config.NUM_CLASSES,
            pretrained=config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "deeplab_mobilenet":
        from models.torchvision_segmentation import create_deeplab_mobilenet

        return create_deeplab_mobilenet(
            num_classes=config.NUM_CLASSES,
            pretrained=config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "monai_unet":
        from models.monai_models import create_monai_unet

        return create_monai_unet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
        )

    if name == "monai_basic_unet":
        from models.monai_models import create_monai_basic_unet

        return create_monai_basic_unet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
        )

    if name == "monai_segresnet":
        from models.monai_models import create_monai_segresnet

        return create_monai_segresnet(
            num_classes=config.NUM_CLASSES,
            initial_filters=config.BASE_CHANNELS,
        )

    if name == "monai_flexible_unet_b0":
        from models.monai_models import create_monai_flexible_unet_b0

        return create_monai_flexible_unet_b0(
            num_classes=config.NUM_CLASSES,
            pretrained=config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "monai_flexible_unet_b1":
        from models.monai_models import create_monai_flexible_unet_b1

        return create_monai_flexible_unet_b1(
            num_classes=config.NUM_CLASSES,
            pretrained=config.USE_PRETRAINED_WEIGHTS,
        )

    raise ValueError(
        f"Unknown MODEL_NAME: {name}. "
        f"Available models: {AVAILABLE_MODELS}"
    )
