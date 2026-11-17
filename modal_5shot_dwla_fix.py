import csv
import json
import os
from typing import List, Tuple

import modal

APP_NAME = "abcdfss-plus-5shot-dwla-fix"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS: List[str] = ["deepglobe", "isic", "lung", "fss", "suim"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "pillow",
        "scipy",
        "scikit-image",
        "kaggle",
        "six",
    )
    .run_commands(
        "python -c 'from torchvision.models import resnet; resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT)'"
    )
    .add_local_dir(
        os.path.join(ROOT_DIR, "ABCDFSS"),
        remote_path="/workspace/ABCDFSS",
        ignore=["__pycache__", ".git", "logs"],
    )
    .add_local_dir(
        os.path.join(ROOT_DIR, "abcdfss_plus"),
        remote_path="/workspace/abcdfss_plus",
        ignore=["__pycache__"],
    )
    .add_local_file(
        os.path.join(ROOT_DIR, "improved_abcdfss.py"),
        remote_path="/workspace/improved_abcdfss.py",
    )
)

datasets_vol = modal.Volume.from_name("abcdfss-datasets", create_if_missing=True)
results_vol = modal.Volume.from_name("abcdfss-results", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="T4",
    volumes={"/data": datasets_vol, "/results": results_vol},
    timeout=43200,
)
def run_unit_dwla_fix(
    benchmark: str,
    shot: int = 5,
    suim_max_episodes: int = 1650,
    seed: int = 42,
    num_epochs: int = 25,
) -> dict:
    import sys

    for p in ("/workspace", "/workspace/ABCDFSS"):
        if p not in sys.path:
            sys.path.insert(0, p)

    from abcdfss_plus.benchmark import run_benchmark

    unit_out = f"/results/table4_dwla_fix/{benchmark}_{shot}shot"
    existing_json = f"{unit_out}/results.json"
    results_vol.reload()
    if os.path.isfile(existing_json):
        with open(existing_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("results"):
            print(f"[skip] table4_dwla_fix/{benchmark}_{shot}shot already completed", flush=True)
            return data["results"][0]

    roots = {k: f"/data/{k}" for k in BENCHMARKS}
    res = run_benchmark(
        dataset_roots=roots,
        output_dir=unit_out,
        benchmarks=[benchmark],
        shots=[shot],
        suim_max_episodes=suim_max_episodes,
        seed=seed,
        num_epochs=num_epochs,
        fit_every_episode=True,
        abcdfss_dir="/workspace/ABCDFSS",
        require_cuda=True,
        checkpoint_every=50,
        on_checkpoint=lambda: results_vol.commit(),
    )
    results_vol.commit()
    return res["results"][0]


@app.function(volumes={"/results": results_vol}, timeout=43200)
def run_part2_5shot_dwla_fix() -> dict:
    unit_pairs: List[Tuple[str, int]] = [("lung", 5), ("isic", 5)]
    units = [(b, s, 1650, 42, 25) for b, s in unit_pairs]
    rows = list(run_unit_dwla_fix.starmap(units))
    os.makedirs("/results/table4_dwla_fix", exist_ok=True)
    payload = {
        "method": "ABCDFSS-Plus (DWLA 5-shot fix)",
        "table": "Table 4 (no-pp)",
        "part": 2,
        "platform": "Modal (T4)",
        "inference_mode": "every-episode",
        "results": rows,
    }
    with open("/results/table4_dwla_fix/part2_5shot.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    return payload
