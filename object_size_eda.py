from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import config

OUTPUT_DIR = config.OUTPUT_DIR / "object_size_eda"
PRIMARY_CLASSES = {"ptm", "nptm"}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def components(mask):
    """8-connected component areas; no scipy/opencv dependency."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    areas = []
    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        area = 0
        while stack:
            y, x = stack.pop()
            area += 1
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and mask[ny, nx]
                        and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        areas.append(area)
    return sorted(areas, reverse=True)

def mask_index(folder):
    return {p.stem: p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    splits = [config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT]

    for split in splits:
        for cls in config.CLASS_DEFINITIONS:
            if cls["name"] not in PRIMARY_CLASSES:
                continue

            folder = config.ANNOTATIONS_DIR / cls["folder"] / split
            masks = mask_index(folder)

            for stem, path in masks.items():
                mask = np.asarray(Image.open(path).convert("L")) > 0
                image_area = mask.size
                fg_area = int(mask.sum())
                sizes = components(mask)

                rows.append({
                    "split": split,
                    "image": stem,
                    "class": cls["name"],
                    "mask_area_px": fg_area,
                    "mask_area_pct_image": 100 * fg_area / image_area,
                    "n_components": len(sizes),
                    "largest_component_px": sizes[0] if sizes else 0,
                    "second_component_px": sizes[1] if len(sizes) > 1 else 0,
                    "largest_component_pct_of_mask":
                        100 * sizes[0] / fg_area if fg_area else 0,
                    "smallest_component_px": sizes[-1] if sizes else 0,
                    "component_sizes_px": ";".join(map(str, sizes)),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No PTM/NPTM masks found.")

    df.to_csv(OUTPUT_DIR / "mask_object_sizes.csv", index=False)

    print("\n=== OBJECT SIZE / FRAGMENTATION EDA ===")
    for name, g in df.groupby("class"):
        nonempty = g[g.mask_area_px > 0]
        print(f"\n{name.upper()}")
        print(f"masks: {len(g)}")
        print(f"empty: {(g.mask_area_px == 0).sum()}")
        print(f"median mask area: {nonempty.mask_area_px.median():.0f} px")
        print(f"median image coverage: {nonempty.mask_area_pct_image.median():.3f}%")
        print(f"median components/mask: {nonempty.n_components.median():.1f}")
        print(f">1 component: {(nonempty.n_components > 1).mean()*100:.1f}%")
        print(f">2 components: {(nonempty.n_components > 2).mean()*100:.1f}%")
        print("median largest-component share: "
              f"{nonempty.largest_component_pct_of_mask.median():.1f}%")

    # Components per GT mask
    plt.figure(figsize=(8, 5))
    names = list(df["class"].unique())
    vals = [df.loc[df["class"] == n, "n_components"] for n in names]
    plt.boxplot(vals, tick_labels=[n.upper() for n in names])
    plt.ylabel("Connected components")
    plt.title("Ground-truth fragmentation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "components_per_mask.png", dpi=250)
    plt.close()

    # Total object size
    plt.figure(figsize=(8, 5))
    for name, g in df.groupby("class"):
        v = g.loc[g.mask_area_px > 0, "mask_area_px"]
        plt.hist(np.log10(v), bins=30, alpha=0.5, label=name.upper())
    plt.xlabel("log10(mask area in pixels)")
    plt.ylabel("Masks")
    plt.title("Ground-truth object-size distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "object_size_distribution.png", dpi=250)
    plt.close()

    # Dominance of largest component
    plt.figure(figsize=(8, 5))
    vals = [
        df.loc[(df["class"] == n) & (df.mask_area_px > 0),
               "largest_component_pct_of_mask"]
        for n in names
    ]
    plt.boxplot(vals, tick_labels=[n.upper() for n in names])
    plt.ylabel("Largest component / total mask area (%)")
    plt.title("Is the GT normally one contiguous object?")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "largest_component_share.png", dpi=250)
    plt.close()

    print(f"\nSaved CSV and plots to: {OUTPUT_DIR}")
    print("\nRule of thumb:")
    print("- largest component usually ~95-100% -> removing small islands is reasonable")
    print("- several substantial GT components -> do NOT blindly keep only the largest")
    print("- inspect component_sizes_px in the CSV before choosing a pixel threshold")

if __name__ == "__main__":
    main()
