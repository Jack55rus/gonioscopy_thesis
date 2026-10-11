import base64, io, os, platform, time
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.morphology import remove_small_objects
import config
from dataset import TrabecularMeshworkDataset
from model_forward import model_forward
from models.factory import create_model

ENSEMBLE_MODELS = [
    {"name": "monai_flexible_unet_b1", "checkpoint": config.OUTPUT_DIR / "monai_flexible_unet_b1_best.pt"},
    {"name": "monai_flexible_unet_b4", "checkpoint": config.OUTPUT_DIR / "monai_flexible_unet_b4_best.pt"},
]
TTA_MODES = ["none", "hvflip"]
NUMBER_OF_SAMPLES = 40
WARMUP_SAMPLES = 5
TM_CLASS_ID = 1
TM_THRESHOLD = 0.5
SPLIT = config.TEST_SPLIT
REPORT_PATH = config.OUTPUT_DIR / "ensemble_benchmark.html"
TEMPLATE_PATH = Path(__file__).with_name("benchmark_report.html")
STAGES = ["loading", "transfer", "tta", "inference", "softmax_unflip", "tta_mean", "ensemble_mean", "threshold", "postprocess", "evaluation"]

def flip(x, mode):
    if mode == "none": return x
    if mode == "hflip": return x.flip([3])
    if mode == "vflip": return x.flip([2])
    if mode == "hvflip": return x.flip([2, 3])
    raise ValueError(mode)

def sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)

def timed(device, func):
    sync(device)
    start = time.perf_counter()
    result = func()
    sync(device)
    return result, (time.perf_counter() - start) * 1000

def load_models(device):
    original = config.MODEL_NAME
    models = []
    try:
        for spec in ENSEMBLE_MODELS:
            config.MODEL_NAME = spec["name"]
            model = create_model().to(device)
            checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            models.append(model)
    finally:
        config.MODEL_NAME = original
    return models

def score(pred, target):
    valid = target != config.IGNORE_INDEX
    gt = (target == TM_CLASS_ID) & valid
    pred = pred & valid
    return np.array([(pred & gt).sum(), (pred & ~gt & valid).sum(), (~pred & gt & valid).sum(), (~pred & ~gt & valid).sum()], dtype=np.int64)

def quality(counts):
    tp, fp, fn, tn = map(int, counts)
    return {"Dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            "IoU": tp / (tp + fp + fn) if tp + fp + fn else 1.,
            "Precision": tp / (tp + fp) if tp + fp else 0.,
            "Recall": tp / (tp + fn) if tp + fn else 0.,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}

def benchmark(device, dataset):
    models = load_models(device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    iterator = iter(loader)
    records, counts = [], np.zeros(4, dtype=np.int64)
    with torch.inference_mode():
        for i in range(WARMUP_SAMPLES + NUMBER_OF_SAMPLES):
            start = time.perf_counter()
            try: batch = next(iterator)
            except StopIteration: break
            stages = {s: 0. for s in STAGES}
            stages["loading"] = (time.perf_counter() - start) * 1000
            image, stages["transfer"] = timed(device, lambda: batch["image"].to(device))
            device_batch = dict(batch)
            per_model = []
            for model in models:
                per_tta = []
                for mode in TTA_MODES:
                    x, ms = timed(device, lambda: flip(image, mode))
                    stages["tta"] += ms
                    device_batch["image"] = x
                    logits, ms = timed(device, lambda: model_forward(model, device_batch, device))
                    stages["inference"] += ms
                    prob, ms = timed(device, lambda: torch.softmax(flip(logits, mode), dim=1))
                    stages["softmax_unflip"] += ms
                    per_tta.append(prob)
                merged, ms = timed(device, lambda: torch.stack(per_tta).mean(0))
                stages["tta_mean"] += ms
                per_model.append(merged)
            probs, stages["ensemble_mean"] = timed(device, lambda: torch.stack(per_model).mean(0))
            pred, stages["threshold"] = timed(device, lambda: (probs[0, TM_CLASS_ID] > TM_THRESHOLD).cpu().numpy())
            pred, stages["postprocess"] = timed(device, lambda: remove_small_objects(pred, min_size=config.POSTPROCESS_MIN_OBJECT_SIZE, connectivity=2))
            c, stages["evaluation"] = timed(device, lambda: score(pred, batch["target"][0].numpy()))
            if i < WARMUP_SAMPLES: continue
            counts += c
            stages["total"] = sum(stages.values())
            records.append(stages)
    if not records: raise RuntimeError("No measured images; reduce WARMUP_SAMPLES.")
    summary = {k: {"mean": float(np.mean([r[k] for r in records])), "p95": float(np.percentile([r[k] for r in records], 95)), "median": float(np.median([r[k] for r in records]))} for k in STAGES + ["total"]}
    memory = None
    if device.type == "cuda":
        try:
            memory = torch.cuda.max_memory_allocated() / 1048576
        except RuntimeError:
            pass
    return {"device": device.type.upper(), "summary": summary, "quality": quality(counts), "n": len(records), "records": records, "peak_gpu_mib": memory}

def image_html(draw):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    draw(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '" style="width:100%">'

def machine_info():
    data = {"Operating system": platform.platform(), "CPU": platform.processor() or platform.machine(),
            "Logical CPU cores": os.cpu_count(), "PyTorch": torch.__version__, "Python": platform.python_version(),
            "CUDA version": torch.version.cuda or "Unavailable"}
    try:
        import psutil
        data["Physical CPU cores"] = psutil.cpu_count(logical=False)
        data["RAM"] = f"{psutil.virtual_memory().total / 1073741824:.1f} GiB"
    except ImportError: pass
    if torch.cuda.is_available():
        data["GPU"] = torch.cuda.get_device_name(0)
        data["GPU memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1073741824:.1f} GiB"
    return data

def chart_breakdown(results):
    def draw(ax):
        y = np.arange(len(STAGES))
        width = 0.76 / len(results)
        for i, result in enumerate(results):
            values = [result["summary"][stage]["mean"] for stage in STAGES]
            ax.barh(y + i * width, values, height=width, label=result["device"])
        ax.set_yticks(y + width * (len(results) - 1) / 2, [stage.replace("_", " ").title() for stage in STAGES])
        ax.invert_yaxis()
        ax.set_xlabel("Mean time per image (ms)")
        ax.legend(frameon=False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    return image_html(draw)

def chart_latency(results):
    def draw(ax):
        for result in results:
            times = np.sort([item["total"] for item in result["records"]])
            ax.plot(np.linspace(0, 100, len(times)), times, linewidth=2.5, label=result["device"])
        ax.set_xlabel("Percentage of images")
        ax.set_ylabel("Time per image (ms)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
    return image_html(draw)

def chart_throughput(results):
    def draw(ax):
        speeds = [1000 / result["summary"]["total"]["mean"] for result in results]
        bars = ax.bar([result["device"] for result in results], speeds, width=0.52)
        ax.set_ylabel("Images per second")
        ax.set_ylim(0, max(speeds) * 1.25)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=12)
    return image_html(draw)

STAGE_DESCRIPTIONS = {
    "loading": "Read the image and apply dataset preprocessing.",
    "transfer": "Move the image tensor to the CPU or GPU.",
    "tta": "Create flipped versions of the input image.",
    "inference": "Run all neural networks on every selected image orientation.",
    "softmax_unflip": "Align the predictions and convert scores into probabilities.",
    "tta_mean": "Average the predictions from the different orientations.",
    "ensemble_mean": "Combine predictions from the selected networks.",
    "threshold": "Convert TM probabilities into a binary prediction mask.",
    "postprocess": "Remove predicted objects smaller than the minimum size.",
    "evaluation": "Compare predictions against annotated masks (benchmark only).",
}

def report(results):
    environment = Environment(loader=FileSystemLoader(str(TEMPLATE_PATH.parent)), autoescape=select_autoescape(["html"]))
    template = environment.get_template(TEMPLATE_PATH.name)
    cpu = next((result for result in results if result["device"] == "CPU"), None)
    gpu = next((result for result in results if result["device"] == "CUDA"), None)
    speedup = cpu["summary"]["total"]["mean"] / gpu["summary"]["total"]["mean"] if cpu and gpu else None
    return template.render(
        generated=datetime.now().strftime("%d %B %Y, %H:%M"),
        results=results, machine=machine_info(), stages=STAGES,
        descriptions=STAGE_DESCRIPTIONS, models=ENSEMBLE_MODELS,
        tta_modes=TTA_MODES, split=SPLIT, threshold=TM_THRESHOLD,
        min_size=config.POSTPROCESS_MIN_OBJECT_SIZE, warmup=WARMUP_SAMPLES,
        speedup=speedup, throughput_chart=chart_throughput(results),
        latency_chart=chart_latency(results), breakdown_chart=chart_breakdown(results),
    )

def main():
    if not ENSEMBLE_MODELS or not TTA_MODES:
        raise ValueError("Specify at least one model and TTA mode.")
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"HTML template missing: {TEMPLATE_PATH}")
    dataset = TrabecularMeshworkDataset(split=SPLIT, training=False)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    results = []
    for device in devices:
        print(f"Benchmarking {device}...")
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass
        result = benchmark(device, dataset)
        results.append(result)
        print(f"{result['device']}: {result['summary']['total']['mean']:.1f} ms/image, Dice={result['quality']['Dice']:.4f}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report(results), encoding="utf-8")
    print(f"Report saved: {REPORT_PATH}")

if __name__ == "__main__":
    main()
