from torch import nn

from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)


class TorchvisionSegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def create_lraspp(num_classes, pretrained=False):
    if pretrained:
        model = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        )

        model.classifier.low_classifier = nn.Conv2d(
            model.classifier.low_classifier.in_channels,
            num_classes,
            kernel_size=1,
        )

        model.classifier.high_classifier = nn.Conv2d(
            model.classifier.high_classifier.in_channels,
            num_classes,
            kernel_size=1,
        )
    else:
        model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    return TorchvisionSegmentationWrapper(model)


def create_deeplab_mobilenet(num_classes, pretrained=False):
    if pretrained:
        model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        )

        final_layer = model.classifier[-1]

        model.classifier[-1] = nn.Conv2d(
            final_layer.in_channels,
            num_classes,
            kernel_size=1,
        )

        model.aux_classifier = None
    else:
        model = deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=False,
        )

    return TorchvisionSegmentationWrapper(model)
