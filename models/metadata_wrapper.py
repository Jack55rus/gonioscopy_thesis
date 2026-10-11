from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MetadataSegmentationWrapper(nn.Module):
    """
    Adds a small metadata-conditioned residual refinement head around any
    existing segmentation network.

    This keeps all current MONAI/custom/torchvision models usable.

    The base network still performs the segmentation. The metadata branch
    conditions a small spatial refinement head using:
      - base logits
      - normalized x/y coordinate maps
      - metadata feature maps

    This is more useful than adding metadata only as a class bias because
    FaceXX contains spatial information.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        metadata_dim: int,
        hidden_channels: int = 32,
    ):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        self.refiner = nn.Sequential(
            nn.Conv2d(
                num_classes + 16 + 2,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                num_classes,
                kernel_size=1,
            ),
        )

        # Start as the original model. The metadata refiner is learned
        # gradually rather than destroying pretrained predictions at step 1.
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(
        self,
        images: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.base_model(images)

        # Allows old inference code to continue working while migrating.
        if metadata is None:
            return logits

        batch_size, _, height, width = logits.shape

        encoded = self.metadata_encoder(metadata)
        encoded = encoded[:, :, None, None].expand(
            -1, -1, height, width
        )

        y = torch.linspace(
            -1.0, 1.0, height,
            device=logits.device,
            dtype=logits.dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, width,
            device=logits.device,
            dtype=logits.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=0)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

        refinement = self.refiner(
            torch.cat((logits, encoded, coords), dim=1)
        )

        return logits + refinement
