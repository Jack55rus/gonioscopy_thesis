import platform
import statistics
import time

import torch

import config
from models.factory import create_model
from utils.common import count_parameters, save_json


CPU_THREAD_COUNTS = [1, 2, 4]
WARMUP_RUNS = 20
MEASURED_RUNS = 100


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


@torch.inference_mode()
def measure(model, device, cpu_threads=None):
    if device.type == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)

    model = model.to(device)
    model.eval()

    example = torch.randn(
        1,
        3,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        device=device,
    )

    for _ in range(WARMUP_RUNS):
        model(example)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []

    for _ in range(MEASURED_RUNS):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        model(example)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        times_ms.append(elapsed_ms)

    median_ms = statistics.median(times_ms)

    return {
        "device": device.type,
        "cpu_threads": cpu_threads,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": median_ms,
        "p90_ms": percentile(times_ms, 0.90),
        "p95_ms": percentile(times_ms, 0.95),
        "images_per_second": 1000.0 / median_ms,
    }


def load_model():
    model = create_model()

    checkpoint = torch.load(
        config.CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])
    return model


def main():
    results = {
        "model_name": config.MODEL_NAME,
        "parameters": count_parameters(create_model()),
        "input_size": [
            1,
            3,
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "measurements": [],
    }

    for thread_count in CPU_THREAD_COUNTS:
        model = load_model()

        result = measure(
            model=model,
            device=torch.device("cpu"),
            cpu_threads=thread_count,
        )

        results["measurements"].append(result)
        print(result)

    if torch.cuda.is_available():
        model = load_model()

        result = measure(
            model=model,
            device=torch.device("cuda"),
        )

        result["gpu_name"] = torch.cuda.get_device_name(0)

        results["measurements"].append(result)
        print(result)

    save_json(results, config.BENCHMARK_PATH)
    print(f"Saved: {config.BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
