import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch

import config
from models.factory import create_model
from utils.common import count_parameters


MODELS_TO_COMPARE = [
    {
        "name": "tiny_unet",
        "base_channels": 16,
    },
    {
        "name": "depthwise_unet",
        "base_channels": 16,
    },
    {
        "name": "unet",
        "base_channels": 32,
    },
    {
        "name": "monai_segresnet",
        "base_channels": 16,
    },
    {
        "name": "monai_flexible_unet_b0",
        "base_channels": 16,
    },
    {
        "name": "monai_flexible_unet_b1",
        "base_channels": 16,
    },
    {
        "name": "lraspp_mobilenet",
        "base_channels": 16,
    },
    {
        "name": "fast_scnn",
        "base_channels": 16,
    },
]


def benchmark_untrained_model(model, device, runs=50):
    model = model.to(device)
    model.eval()

    sample = torch.randn(
        1,
        3,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(10):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append(
                (time.perf_counter() - start) * 1000.0
            )

    return sum(times) / len(times)


def main():
    original_model_name = config.MODEL_NAME
    original_base_channels = config.BASE_CHANNELS

    rows = []

    for model_config in MODELS_TO_COMPARE:
        print(f'model: {model_config["name"]}')
        config.MODEL_NAME = model_config["name"]
        config.BASE_CHANNELS = model_config["base_channels"]

        model = create_model()

        row = {
            "model": config.MODEL_NAME,
            "base_channels": config.BASE_CHANNELS,
            "parameters": count_parameters(model),
        }

        torch.set_num_threads(1)

        row["cpu_1_thread_mean_ms"] = benchmark_untrained_model(
            model=create_model(),
            device=torch.device("cpu"),
        )

        if torch.cuda.is_available():
            row["gpu_mean_ms"] = benchmark_untrained_model(
                model=create_model(),
                device=torch.device("cuda"),
            )

        metrics_path = (
            config.OUTPUT_DIR
            / f"{config.MODEL_NAME}_test_metrics.json"
        )

        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)

            row["test_macro_dice"] = metrics.get(
                # "macro_foreground_dice"
                "primary_mean_dice"
            )

            row["test_ptm_dice"] = metrics.get(
                "ptm", {}
            ).get("dice")

            row["test_nptm_dice"] = metrics.get(
                "nptm", {}
            ).get("dice")

            row["test_merged_tm_dice"] = metrics.get(
                "merged_tm_dice"
            )

        rows.append(row)

    config.MODEL_NAME = original_model_name
    config.BASE_CHANNELS = original_base_channels

    table = pd.DataFrame(rows)

    output_path = config.OUTPUT_DIR / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    print(table)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
