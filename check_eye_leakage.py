from collections import defaultdict

import config
from dataset import find_samples
from metadata_features import parse_filename_metadata


def eye_ids(split):
    return {
        parse_filename_metadata(sample.image_path.stem).eye_id
        for sample in find_samples(split)
    }


def main():
    split_ids = {
        config.TRAIN_SPLIT: eye_ids(config.TRAIN_SPLIT),
        config.VAL_SPLIT: eye_ids(config.VAL_SPLIT),
        config.TEST_SPLIT: eye_ids(config.TEST_SPLIT),
    }

    for split, ids in split_ids.items():
        print(f"{split}: {len(ids)} unique eyes")

    names = list(split_ids)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = split_ids[left] & split_ids[right]
            print(
                f"{left} vs {right}: "
                f"{len(overlap)} overlapping eyes"
            )
            if overlap:
                print("  ", sorted(overlap))


if __name__ == "__main__":
    main()
