
import csv
import json
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")
OUTPUT_FILE = Path("results/metrics_summary.csv")

rows = []

for file in sorted(EXPERIMENTS_DIR.glob("exp*/results.json")):
    data = json.loads(file.read_text(encoding="utf-8"))

    hp = data.get("hyperparameters", {})
    dataset = data.get("dataset", {})

    rows.append({
        "experiment": data["experiment"],
        "dataset_version": data.get("dataset_version", "v1"),
        "epochs": hp.get("epochs", data.get("epochs")),
        "learning_rate": hp.get(
            "learning_rate", data.get("learning_rate")
        ),
        "lora_r": hp.get("lora_r", data.get("lora_r")),
        "lora_alpha": hp.get(
            "lora_alpha", data.get("lora_alpha")
        ),
        "train_samples": dataset.get(
            "train_samples", data.get("train_samples")
        ),
        "validation_samples": dataset.get(
            "validation_samples", data.get("validation_samples")
        ),
        "train_loss": round(
            data["train_metrics"]["train_loss"], 4
        ),
        "validation_loss": round(
            data["eval_metrics"]["eval_loss"], 4
        ),
        "training_time_seconds": round(
            data["training_time_seconds"], 2
        ),
        "peak_gpu_memory_gb": round(
            data["peak_gpu_memory_gb"], 2
        ),
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Arquivo criado: {OUTPUT_FILE}")
print(f"Experimentos exportados: {len(rows)}")
