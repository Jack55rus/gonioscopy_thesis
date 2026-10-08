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
    "monai_flexible_unet_b4",
    "monai_flexible_unet_b7",
    "monai_swin_unetr",
]


def _create_base_model():
    name = config.MODEL_NAME

    if name == "monai_swin_unetr":
        from models.monai_models import create_monai_swin_unetr
        return create_monai_swin_unetr(config.NUM_CLASSES)

    if name == "tiny_unet":
        from models.tiny_unet import TinyUNet
        return TinyUNet(config.NUM_CLASSES, config.BASE_CHANNELS)

    if name in {"depthwise_unet", "unet"}:
        from models.unet import UNet
        return UNet(
            num_classes=config.NUM_CLASSES,
            base_channels=config.BASE_CHANNELS,
            depthwise=(name == "depthwise_unet"),
        )

    if name == "fast_scnn":
        from models.fast_scnn import FastSCNN
        return FastSCNN(num_classes=config.NUM_CLASSES)

    if name == "lraspp_mobilenet":
        from models.torchvision_segmentation import create_lraspp
        return create_lraspp(
            config.NUM_CLASSES,
            config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "deeplab_mobilenet":
        from models.torchvision_segmentation import create_deeplab_mobilenet
        return create_deeplab_mobilenet(
            config.NUM_CLASSES,
            config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "monai_unet":
        from models.monai_models import create_monai_unet
        return create_monai_unet(config.NUM_CLASSES, config.BASE_CHANNELS)

    if name == "monai_basic_unet":
        from models.monai_models import create_monai_basic_unet
        return create_monai_basic_unet(config.NUM_CLASSES, config.BASE_CHANNELS)

    if name == "monai_segresnet":
        from models.monai_models import create_monai_segresnet
        return create_monai_segresnet(config.NUM_CLASSES, config.BASE_CHANNELS)

    if name == "monai_flexible_unet_b0":
        from models.monai_models import create_monai_flexible_unet_b0
        return create_monai_flexible_unet_b0(
            config.NUM_CLASSES,
            config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "monai_flexible_unet_b1":
        from models.monai_models import create_monai_flexible_unet_b1
        return create_monai_flexible_unet_b1(
            config.NUM_CLASSES,
            config.USE_PRETRAINED_WEIGHTS,
        )

    if name == "monai_flexible_unet_b4":
        from models.monai_models import create_monai_flexible_unet_b4
        return create_monai_flexible_unet_b4(
            config.NUM_CLASSES,
            config.USE_PRETRAINED_WEIGHTS,
        )
    if name == "monai_flexible_unet_b7":
        from models.monai_models import create_monai_flexible_unet_b7
        return create_monai_flexible_unet_b7(
            config.NUM_CLASSES,
            config.USE_PRETRAINED_WEIGHTS,
        )

    raise ValueError(
        f"Unknown MODEL_NAME: {name}. Available: {AVAILABLE_MODELS}"
    )


def create_model():
    model = _create_base_model()

    if not config.USE_METADATA:
        return model

    from metadata_features import METADATA_DIM
    from models.metadata_wrapper import MetadataSegmentationWrapper

    return MetadataSegmentationWrapper(
        base_model=model,
        num_classes=config.NUM_CLASSES,
        metadata_dim=METADATA_DIM,
        hidden_channels=config.METADATA_REFINER_CHANNELS,
    )
