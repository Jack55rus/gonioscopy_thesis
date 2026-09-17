from monai.networks.nets import BasicUNet, FlexibleUNet, SegResNet, UNet


def create_monai_unet(
    num_classes: int,
    base_channels: int = 16,
):
    return UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        channels=(
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="BATCH",
        dropout=0.0,
    )


def create_monai_basic_unet(
    num_classes: int,
    base_channels: int = 16,
):
    return BasicUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        features=(
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
            base_channels,
        ),
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("batch", {"affine": True}),
        dropout=0.0,
        upsample="deconv",
    )


def create_monai_segresnet(
    num_classes: int,
    initial_filters: int = 16,
):
    return SegResNet(
        spatial_dims=2,
        init_filters=initial_filters,
        in_channels=3,
        out_channels=num_classes,
        dropout_prob=0.0,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        norm="BATCH",
        act="RELU",
        upsample_mode="nontrainable",
        use_conv_final=True,
    )


def create_monai_flexible_unet_b0(
    num_classes: int,
    pretrained: bool = False,
):
    return FlexibleUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        backbone="efficientnet-b0",
        pretrained=pretrained,
        decoder_channels=(128, 64, 32, 16, 8),
        norm=("batch", {"eps": 1e-3, "momentum": 0.1}),
        act=("relu", {"inplace": True}),
        dropout=0.0,
        upsample="nontrainable",
        interp_mode="bilinear",
        is_pad=True,
    )


def create_monai_flexible_unet_b1(
    num_classes: int,
    pretrained: bool = False,
):
    return FlexibleUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        backbone="efficientnet-b1",
        pretrained=pretrained,
        decoder_channels=(160, 80, 40, 20, 10),
        norm=("batch", {"eps": 1e-3, "momentum": 0.1}),
        act=("relu", {"inplace": True}),
        dropout=0.0,
        upsample="nontrainable",
        interp_mode="bilinear",
        is_pad=True,
    )
