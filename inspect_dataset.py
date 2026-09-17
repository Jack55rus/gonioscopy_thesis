import numpy as np
from PIL import Image

import config
from dataset import find_samples


def inspect_split(split):
    samples = find_samples(split)

    overlap_pixels = 0
    size_mismatches = []
    non_binary_masks = []

    for sample in samples:
        image = Image.open(sample.image_path)
        ptm_mask = Image.open(sample.ptm_mask_path).convert("L")
        nptm_mask = Image.open(sample.nptm_mask_path).convert("L")

        if image.size != ptm_mask.size or image.size != nptm_mask.size:
            size_mismatches.append(
                {
                    "image": sample.image_path.name,
                    "image_size": image.size,
                    "ptm_size": ptm_mask.size,
                    "nptm_size": nptm_mask.size,
                }
            )
            continue

        ptm = np.asarray(ptm_mask)
        nptm = np.asarray(nptm_mask)

        ptm_values = np.unique(ptm)
        nptm_values = np.unique(nptm)

        if not set(ptm_values.tolist()).issubset({0, 1, 255}):
            non_binary_masks.append(
                {
                    "mask": sample.ptm_mask_path.name,
                    "values": ptm_values[:20].tolist(),
                }
            )

        if not set(nptm_values.tolist()).issubset({0, 1, 255}):
            non_binary_masks.append(
                {
                    "mask": sample.nptm_mask_path.name,
                    "values": nptm_values[:20].tolist(),
                }
            )

        overlap_pixels += int(
            ((ptm > 0) & (nptm > 0)).sum()
        )

    print(f"\nSplit: {split}")
    print(f"Complete samples: {len(samples)}")
    print(f"Size mismatches: {len(size_mismatches)}")
    print(f"Non-binary masks: {len(non_binary_masks)}")
    print(f"PTM/NPTM overlapping pixels: {overlap_pixels}")

    if size_mismatches:
        print("\nFirst size mismatches:")
        for item in size_mismatches[:10]:
            print(item)

    if non_binary_masks:
        print("\nFirst non-binary masks:")
        for item in non_binary_masks[:10]:
            print(item)


def main():
    inspect_split(config.TRAIN_SPLIT)
    inspect_split(config.VAL_SPLIT)
    inspect_split(config.TEST_SPLIT)


if __name__ == "__main__":
    main()
