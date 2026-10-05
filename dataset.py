from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

import config
from metadata_features import encode_metadata, parse_filename_metadata


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
}


@dataclass
class Sample:
    image_path: Path
    mask_paths: list[Path]


def files_by_stem(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def find_samples(split: str) -> list[Sample]:
    image_folder = config.ORIGINALS_DIR / split
    images = files_by_stem(image_folder)

    class_mask_files = []
    for class_definition in config.CLASS_DEFINITIONS:
        folder = (
            config.ANNOTATIONS_DIR
            / class_definition["folder"]
            / split
        )
        class_mask_files.append(files_by_stem(folder))

    samples = []

    for image_stem, image_path in sorted(images.items()):
        mask_paths = []
        complete = True

        for class_definition, mask_files in zip(
            config.CLASS_DEFINITIONS,
            class_mask_files,
        ):
            expected_stem = image_stem + class_definition["suffix"]
            mask_path = mask_files.get(expected_stem)

            if mask_path is None:
                complete = False
                break

            mask_paths.append(mask_path)

        if not complete:
            if config.SKIP_INCOMPLETE_SAMPLES:
                continue
            raise FileNotFoundError(
                f"One or more masks are missing for {image_path.name}"
            )

        samples.append(
            Sample(
                image_path=image_path,
                mask_paths=mask_paths,
            )
        )

    if not samples:
        raise RuntimeError(f"No complete samples found in split: {split}")

    return samples


class JointTransform:
    def __init__(self, training: bool):
        self.training = training

    def _geometric_transform(self, image, masks):
        # Horizontal flip remains configurable. If later you establish that
        # left/right orientation carries fixed clinical meaning, disable it.
        if random.random() < config.HORIZONTAL_FLIP_PROBABILITY:
            image = TF.hflip(image)
            masks = [TF.hflip(mask) for mask in masks]

        angle = random.uniform(
            -config.MAX_ROTATION_DEGREES,
            config.MAX_ROTATION_DEGREES,
        )

        # Small translations/scaling are useful for limited data while
        # preserving the global acquisition sector.
        translate = [
            int(random.uniform(-1, 1) * config.MAX_TRANSLATE_FRACTION * image.width),
            int(random.uniform(-1, 1) * config.MAX_TRANSLATE_FRACTION * image.height),
        ]
        scale = random.uniform(
            1.0 - config.MAX_SCALE_JITTER,
            1.0 + config.MAX_SCALE_JITTER,
        )

        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )

        masks = [
            TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            for mask in masks
        ]

        return image, masks

    def _photometric_transform(self, image):
        brightness = random.uniform(
            1.0 - config.BRIGHTNESS_JITTER,
            1.0 + config.BRIGHTNESS_JITTER,
        )
        contrast = random.uniform(
            1.0 - config.CONTRAST_JITTER,
            1.0 + config.CONTRAST_JITTER,
        )
        saturation = random.uniform(
            1.0 - config.SATURATION_JITTER,
            1.0 + config.SATURATION_JITTER,
        )

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)

        if random.random() < config.BLUR_PROBABILITY:
            radius = random.uniform(0.1, config.MAX_BLUR_RADIUS)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        return image

    def __call__(self, image: Image.Image, masks: list[Image.Image]):
        if self.training:
            image, masks = self._geometric_transform(image, masks)
            image = self._photometric_transform(image)

        output_size = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

        image = TF.resize(
            image,
            output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        masks = [
            TF.resize(
                mask,
                output_size,
                interpolation=InterpolationMode.NEAREST,
            )
            for mask in masks
        ]

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        binary_masks = [
            np.asarray(mask) > 0
            for mask in masks
        ]

        # Current project setting: PTM + NPTM -> one TM class.
        tm_mask = np.logical_or(binary_masks[0], binary_masks[1])

        target = np.zeros(tm_mask.shape, dtype=np.uint8)
        target[tm_mask] = 1

        return image, torch.from_numpy(target.astype(np.int64))


class TrabecularMeshworkDataset(Dataset):
    CLASS_NAMES = config.CLASS_NAMES

    def __init__(self, split: str, training: bool):
        self.samples = find_samples(split)
        self.transform = JointTransform(training=training)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        masks = [
            Image.open(mask_path).convert("L")
            for mask_path in sample.mask_paths
        ]

        for mask_path, mask in zip(sample.mask_paths, masks):
            if image.size != mask.size:
                raise ValueError(
                    f"Size mismatch for {sample.image_path.name}: "
                    f"image={image.size}, mask={mask_path.name}, "
                    f"mask size={mask.size}"
                )

        image, target = self.transform(image=image, masks=masks)

        parsed = parse_filename_metadata(sample.image_path.stem)
        metadata = encode_metadata(
            parsed,
            number_of_faces=config.NUMBER_OF_FACES,
            max_shot=config.MAX_SHOT_NUMBER,
        )

        return {
            "image": image,
            "target": target,
            "metadata": metadata,
            "eye_id": parsed.eye_id,
            "face": parsed.face,
            "shot": parsed.shot,
            "name": sample.image_path.stem,
        }
