from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path("outputs/model_comparison.csv")
OUTPUT_DIR = Path("outputs/figures_only_primary")
DPI = 300


def save(name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run compare_models.py first."
        )

    df = pd.read_csv(CSV_PATH)

    if df.empty:
        raise ValueError("The comparison CSV is empty.")

    # 1. PTM, NPTM and merged-TM Dice
    dice_columns = [
        column
        for column in [
            "test_ptm_dice",
            "test_nptm_dice",
            "test_merged_tm_dice",
        ]
        if column in df.columns
    ]

    if "model" in df.columns and dice_columns:
        plot_df = df.set_index("model")[dice_columns]
        ax = plot_df.plot(kind="bar", figsize=(10, 6))
        ax.set_title("Segmentation Dice scores")
        ax.set_xlabel("Model")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        save("dice_scores.png")

    # 2. Parameter count
    if {"model", "parameters"}.issubset(df.columns):
        plot_df = df[["model", "parameters"]].dropna().sort_values("parameters")

        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["model"], plot_df["parameters"] / 1_000_000)
        plt.title("Trainable parameter count")
        plt.xlabel("Model")
        plt.ylabel("Parameters, millions")
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.3)
        save("parameter_count.png")

    # 3. CPU latency
    cpu_columns = [
        column
        for column in df.columns
        if column.startswith("cpu_")
        and (
            column.endswith("_median_ms")
            or column.endswith("_mean_ms")
        )
    ]

    if "model" in df.columns and cpu_columns:
        plot_df = df.set_index("model")[cpu_columns]
        ax = plot_df.plot(kind="bar", figsize=(10, 6))
        ax.set_title("CPU inference latency")
        ax.set_xlabel("Model")
        ax.set_ylabel("Latency, ms per image")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        save("cpu_latency.png")

    # 4. GPU latency
    gpu_column = first_existing(
        df,
        ["gpu_median_ms", "gpu_mean_ms"],
    )

    if "model" in df.columns and gpu_column:
        plot_df = df[["model", gpu_column]].dropna().sort_values(gpu_column)

        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["model"], plot_df[gpu_column])
        plt.title("GPU inference latency")
        plt.xlabel("Model")
        plt.ylabel("Latency, ms per image")
        plt.xticks(rotation=20)
        plt.grid(axis="y", alpha=0.3)
        save("gpu_latency.png")

    # 5. Accuracy-versus-CPU-speed trade-off
    dice_column = first_existing(
        df,
        ["test_macro_dice", "test_mean_dice"],
    )

    preferred_cpu_column = first_existing(
        df,
        [
            "cpu_1_thread_median_ms",
            "cpu_1_threads_median_ms",
            "cpu_1_thread_mean_ms",
        ],
    )

    if preferred_cpu_column is None and cpu_columns:
        preferred_cpu_column = cpu_columns[0]

    required = {
        "model",
        "parameters",
        dice_column,
        preferred_cpu_column,
    }

    if None not in required and required.issubset(df.columns):
        plot_df = df[
            [
                "model",
                "parameters",
                dice_column,
                preferred_cpu_column,
            ]
        ].dropna()

        if not plot_df.empty:
            sizes = (
                plot_df["parameters"]
                / plot_df["parameters"].max()
                * 900
                + 100
            )

            plt.figure(figsize=(9, 6))
            plt.scatter(
                plot_df[preferred_cpu_column],
                plot_df[dice_column],
                s=sizes,
                alpha=0.75,
            )

            for _, row in plot_df.iterrows():
                plt.annotate(
                    row["model"],
                    (
                        row[preferred_cpu_column],
                        row[dice_column],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            plt.title("Accuracy-efficiency trade-off")
            plt.xlabel("CPU latency, ms per image")
            plt.ylabel("Mean foreground Dice")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)
            save("accuracy_efficiency_tradeoff.png")

    print(f"Figures directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
