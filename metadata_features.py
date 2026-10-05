from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch


_FILENAME_RE = re.compile(
    r"^(?P<eye>.+?)_"
    r"(?P<device>GS)-"
    r"Face(?P<face>\d+)-"
    r"Shot(?P<shot>\d+)"
    r"(?P<rest>.*)$"
)


@dataclass(frozen=True)
class ImageMetadata:
    eye_id: str
    device: str
    face: int
    shot: int
    angle_open: float
    direct_illumination: float
    high_power: float


def parse_filename_metadata(stem: str) -> ImageMetadata:
    """
    Example:
        DS01_01_GS-Face04-Shot08_OA_DI_HP

    DS01_01 is retained as eye_id for grouping/leakage checks only.
    It is intentionally NOT encoded as a neural-network feature.
    """
    match = _FILENAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Unexpected gonioscopy filename: {stem}")

    tokens = {
        token
        for token in match.group("rest").strip("_").split("_")
        if token
    }

    return ImageMetadata(
        eye_id=match.group("eye"),
        device=match.group("device"),
        face=int(match.group("face")),
        shot=int(match.group("shot")),
        angle_open=float("OA" in tokens),
        direct_illumination=float("DI" in tokens),
        high_power=float("HP" in tokens),
    )


def encode_metadata(
    metadata: ImageMetadata,
    number_of_faces: int = 16,
    max_shot: int = 16,
) -> torch.Tensor:
    """
    Compact metadata vector.

    Face is circular, therefore sin/cos is preferable to a raw integer:
    Face01 and Face16 remain close in representation.

    Eye ID is deliberately excluded to prevent subject memorization.
    Shot is included only as a weak normalized acquisition feature.
    """
    angle = 2.0 * math.pi * (metadata.face - 1) / number_of_faces

    shot = min(max(metadata.shot, 0), max_shot) / max_shot

    return torch.tensor(
        [
            math.sin(angle),
            math.cos(angle),
            shot,
            metadata.angle_open,
            metadata.direct_illumination,
            metadata.high_power,
        ],
        dtype=torch.float32,
    )


METADATA_DIM = 6
