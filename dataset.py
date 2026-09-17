# from __future__ import annotations
#
# import random
# from dataclasses import dataclass
# from pathlib import Path
#
# import numpy as np
# import torch
# from PIL import Image, ImageEnhance
# from torch.utils.data import Dataset
# from torchvision.transforms import functional as TF
# from torchvision.transforms.functional import InterpolationMode
#
# import config
#
#
# SUPPORTED_EXTENSIONS = {
#     ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"
# }
#
#
# @dataclass
# class Sample:
#     image_path: Path
#     ptm_mask_path: Path
#     nptm_mask_path: Path
#
#
# def files_by_stem(folder: Path) -> dict[str, Path]:
#     result = {}
#
#     if not folder.exists():
#         raise FileNotFoundError(f"Folder does not exist: {folder}")
#
#     for path in folder.iterdir():
#         if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
#             result[path.stem] = path
#
#     return result
#
#
# def find_samples(split: str) -> list[Sample]:
#     image_folder = config.ORIGINALS_DIR / split
#     ptm_folder = config.ANNOTATIONS_DIR / config.PTM_FOLDER / split
#     nptm_folder = config.ANNOTATIONS_DIR / config.NPTM_FOLDER / split
#
#     images = files_by_stem(image_folder)
#     ptm_masks = files_by_stem(ptm_folder)
#     nptm_masks = files_by_stem(nptm_folder)
#
#     samples = []
#
#     for image_stem, image_path in sorted(images.items()):
#         ptm_path = ptm_masks.get(image_stem + config.PTM_SUFFIX)
#         nptm_path = nptm_masks.get(image_stem + config.NPTM_SUFFIX)
#
#         if ptm_path is None or nptm_path is None:
#             if config.SKIP_INCOMPLETE_SAMPLES:
#                 continue
#
#             missing = []
#             if ptm_path is None:
#                 missing.append("PTM")
#             if nptm_path is None:
#                 missing.append("NPTM")
#
#             raise FileNotFoundError(
#                 f"Missing {' and '.join(missing)} mask for {image_path.name}"
#             )
#
#         samples.append(
#             Sample(
#                 image_path=image_path,
#                 ptm_mask_path=ptm_path,
#                 nptm_mask_path=nptm_path,
#             )
#         )
#
#     if not samples:
#         raise RuntimeError(f"No complete samples found in split: {split}")
#
#     return samples
#
#
# class JointTransform:
#     def __init__(self, training: bool):
#         self.training = training
#
#     def __call__(
#         self,
#         image: Image.Image,
#         ptm_mask: Image.Image,
#         nptm_mask: Image.Image,
#     ):
#         if self.training:
#             if random.random() < config.HORIZONTAL_FLIP_PROBABILITY:
#                 image = TF.hflip(image)
#                 ptm_mask = TF.hflip(ptm_mask)
#                 nptm_mask = TF.hflip(nptm_mask)
#
#             angle = random.uniform(
#                 -config.MAX_ROTATION_DEGREES,
#                 config.MAX_ROTATION_DEGREES,
#             )
#
#             image = TF.rotate(
#                 image,
#                 angle,
#                 interpolation=InterpolationMode.BILINEAR,
#                 fill=0,
#             )
#
#             ptm_mask = TF.rotate(
#                 ptm_mask,
#                 angle,
#                 interpolation=InterpolationMode.NEAREST,
#                 fill=0,
#             )
#
#             nptm_mask = TF.rotate(
#                 nptm_mask,
#                 angle,
#                 interpolation=InterpolationMode.NEAREST,
#                 fill=0,
#             )
#
#             brightness_factor = random.uniform(
#                 1.0 - config.BRIGHTNESS_JITTER,
#                 1.0 + config.BRIGHTNESS_JITTER,
#             )
#
#             contrast_factor = random.uniform(
#                 1.0 - config.CONTRAST_JITTER,
#                 1.0 + config.CONTRAST_JITTER,
#             )
#
#             image = ImageEnhance.Brightness(image).enhance(brightness_factor)
#             image = ImageEnhance.Contrast(image).enhance(contrast_factor)
#
#         output_size = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
#
#         image = TF.resize(
#             image,
#             output_size,
#             interpolation=InterpolationMode.BILINEAR,
#             antialias=True,
#         )
#
#         ptm_mask = TF.resize(
#             ptm_mask,
#             output_size,
#             interpolation=InterpolationMode.NEAREST,
#         )
#
#         nptm_mask = TF.resize(
#             nptm_mask,
#             output_size,
#             interpolation=InterpolationMode.NEAREST,
#         )
#
#         image = TF.to_tensor(image)
#         image = TF.normalize(
#             image,
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         )
#
#         ptm = np.asarray(ptm_mask) > 0
#         nptm = np.asarray(nptm_mask) > 0
#
#         target = np.zeros(ptm.shape, dtype=np.uint8)
#         target[ptm] = 1
#         target[nptm] = 2
#
#         overlap = ptm & nptm
#
#         if config.IGNORE_OVERLAPS:
#             target[overlap] = config.IGNORE_INDEX
#
#         return image, torch.from_numpy(target.astype(np.int64))
#
#
# class TrabecularMeshworkDataset(Dataset):
#     CLASS_NAMES = ["background", "ptm", "nptm"]
#
#     def __init__(self, split: str, training: bool):
#         self.samples = find_samples(split)
#         self.transform = JointTransform(training=training)
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, index):
#         sample = self.samples[index]
#
#         image = Image.open(sample.image_path).convert("RGB")
#         ptm_mask = Image.open(sample.ptm_mask_path).convert("L")
#         nptm_mask = Image.open(sample.nptm_mask_path).convert("L")
#
#         if image.size != ptm_mask.size or image.size != nptm_mask.size:
#             raise ValueError(
#                 f"Size mismatch for {sample.image_path.name}: "
#                 f"image={image.size}, "
#                 f"PTM={ptm_mask.size}, "
#                 f"NPTM={nptm_mask.size}"
#             )
#
#         image, target = self.transform(
#             image=image,
#             ptm_mask=ptm_mask,
#             nptm_mask=nptm_mask,
#         )
#
#         return {
#             "image": image,
#             "target": target,
#             "name": sample.image_path.stem,
#         }


from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

import config


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
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
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def find_samples(split: str) -> list[Sample]:
    image_folder = config.ORIGINALS_DIR / split
    images = files_by_stem(image_folder)

    class_mask_files = []

    for class_definition in config.CLASS_DEFINITIONS:
        class_folder = (
            config.ANNOTATIONS_DIR
            / class_definition["folder"]
            / split
        )

        class_mask_files.append(
            files_by_stem(class_folder)
        )

    samples = []

    for image_stem, image_path in sorted(images.items()):
        mask_paths = []
        sample_is_complete = True

        for class_definition, mask_files in zip(
            config.CLASS_DEFINITIONS,
            class_mask_files,
        ):
            expected_stem = (
                image_stem
                + class_definition["suffix"]
            )

            mask_path = mask_files.get(expected_stem)

            if mask_path is None:
                sample_is_complete = False
                break

            mask_paths.append(mask_path)

        if not sample_is_complete:
            if config.SKIP_INCOMPLETE_SAMPLES:
                continue

            raise FileNotFoundError(
                f"One or more masks are missing for "
                f"{image_path.name}"
            )

        samples.append(
            Sample(
                image_path=image_path,
                mask_paths=mask_paths,
            )
        )

    if not samples:
        raise RuntimeError(
            f"No complete samples found in split: {split}"
        )

    return samples


class JointTransform:
    def __init__(self, training: bool):
        self.training = training

    def __call__(
        self,
        image: Image.Image,
        masks: list[Image.Image],
    ):
        if self.training:
            if (
                random.random()
                < config.HORIZONTAL_FLIP_PROBABILITY
            ):
                image = TF.hflip(image)
                masks = [
                    TF.hflip(mask)
                    for mask in masks
                ]

            angle = random.uniform(
                -config.MAX_ROTATION_DEGREES,
                config.MAX_ROTATION_DEGREES,
            )

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            masks = [
                TF.rotate(
                    mask,
                    angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
                for mask in masks
            ]

            brightness_factor = random.uniform(
                1.0 - config.BRIGHTNESS_JITTER,
                1.0 + config.BRIGHTNESS_JITTER,
            )

            contrast_factor = random.uniform(
                1.0 - config.CONTRAST_JITTER,
                1.0 + config.CONTRAST_JITTER,
            )

            image = ImageEnhance.Brightness(
                image
            ).enhance(brightness_factor)

            image = ImageEnhance.Contrast(
                image
            ).enhance(contrast_factor)

        output_size = (
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        )

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

        binary_masks = np.stack(
            [
                np.asarray(mask) > 0
                for mask in masks
            ],
            axis=0,
        )

        height, width = binary_masks.shape[1:]

        target = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        number_of_active_classes = (
            binary_masks.sum(axis=0)
        )

        # Assign class indices 1, 2, 3...
        for class_index, binary_mask in enumerate(
            binary_masks,
            start=1,
        ):
            valid_class_pixels = (
                binary_mask
                & (number_of_active_classes == 1)
            )

            target[valid_class_pixels] = class_index

        # Ignore pixels assigned to multiple structures.
        overlapping_pixels = (
            number_of_active_classes > 1
        )

        if config.IGNORE_OVERLAPS:
            target[
                overlapping_pixels
            ] = config.IGNORE_INDEX

        return (
            image,
            torch.from_numpy(
                target.astype(np.int64)
            ),
        )


class TrabecularMeshworkDataset(Dataset):
    CLASS_NAMES = config.CLASS_NAMES

    def __init__(
        self,
        split: str,
        training: bool,
    ):
        self.samples = find_samples(split)
        self.transform = JointTransform(
            training=training
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(
            sample.image_path
        ).convert("RGB")

        masks = [
            Image.open(mask_path).convert("L")
            for mask_path in sample.mask_paths
        ]

        for mask_path, mask in zip(
            sample.mask_paths,
            masks,
        ):
            if image.size != mask.size:
                raise ValueError(
                    f"Size mismatch for "
                    f"{sample.image_path.name}: "
                    f"image={image.size}, "
                    f"mask={mask_path.name}, "
                    f"mask size={mask.size}"
                )

        image, target = self.transform(
            image=image,
            masks=masks,
        )

        return {
            "image": image,
            "target": target,
            "name": sample.image_path.stem,
        }
