import csv
import json
import os
from typing import Dict, List, Tuple

import modal

APP_NAME = "abcdfss-plus-table4"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_SLUGS: Dict[str, str] = {
    "deepglobe": "heyoujue/deepglobe",
    "isic": "heyoujue/isic2018-classwise",
    "lung": "heyoujue/lungsegmentation",
    "fss": "meowmeowmeowmeowmeow/fss1000-a-1000-class-fewshot-segmentation",
    "suim": "heyoujue/suim-merged",
}

BENCHMARKS: List[str] = ["deepglobe", "isic", "lung", "fss", "suim"]
SHOTS: List[int] = [1, 5]

# Balanced 3-part split for Table 4 (every-episode, no-pp on Modal T4)
PART_UNITS: Dict[int, List[Tuple[str, int]]] = {
    1: [("lung", 1), ("suim", 1), ("fss", 1), ("fss", 5)],
    2: [("deepglobe", 1), ("lung", 5), ("isic", 5)],
    3: [("isic", 1), ("suim", 5), ("deepglobe", 5)],
}

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
    .add_local_dir(os.path.join(ROOT_DIR, "ABCDFSS"), remote_path="/workspace/ABCDFSS", ignore=["__pycache__", ".git", "logs"])
    .add_local_dir(os.path.join(ROOT_DIR, "abcdfss_plus"), remote_path="/workspace/abcdfss_plus", ignore=["__pycache__"])
    .add_local_file(os.path.join(ROOT_DIR, "improved_abcdfss.py"), remote_path="/workspace/improved_abcdfss.py")
)

datasets_vol = modal.Volume.from_name("abcdfss-datasets", create_if_missing=True)
results_vol = modal.Volume.from_name("abcdfss-results", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


def _load_kaggle_auth() -> Dict[str, str]:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return {"KAGGLE_API_TOKEN": os.environ["KAGGLE_API_TOKEN"].strip()}
    token_path = os.path.expanduser(os.path.join("~", ".kaggle", "access_token"))
    if os.path.isfile(token_path):
        with open(token_path, "r", encoding="utf-8") as f:
            return {"KAGGLE_API_TOKEN": f.read().strip()}
    json_path = os.path.expanduser(os.path.join("~", ".kaggle", "kaggle.json"))
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"KAGGLE_USERNAME": data["username"], "KAGGLE_KEY": data["key"]}
    raise FileNotFoundError(
        "Kaggle credentials not found. Set KAGGLE_API_TOKEN or place ~/.kaggle/access_token or ~/.kaggle/kaggle.json."
    )


@app.function(volumes={"/data": datasets_vol}, timeout=3600)
def ensure_datasets(kaggle_env: Dict[str, str], required_keys: List[str]) -> Dict[str, str]:
    import subprocess

    for k, v in kaggle_env.items():
        os.environ[k] = v

    resolved: Dict[str, str] = {}
    for key in required_keys:
        slug = DATASET_SLUGS[key]
        target_dir = f"/data/{key}"
        if os.path.isdir(target_dir) and any(os.scandir(target_dir)):
            print(f"[skip] {key} already present in {target_dir}", flush=True)
            resolved[key] = target_dir
            continue

        os.makedirs(target_dir, exist_ok=True)
        print(f"[download] {key} ({slug}) -> {target_dir}", flush=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", target_dir, "--unzip"],
            check=True,
        )
        datasets_vol.commit()
        resolved[key] = target_dir
    return resolved


@app.function(
    gpu="T4",
    volumes={"/data": datasets_vol, "/results": results_vol},
    timeout=43200,
)
def run_table4_unit(
    benchmark: str,
    shot: int,
    suim_max_episodes: int = 1650,
    seed: int = 42,
    num_epochs: int = 25,
) -> dict:
    import sys

    for p in ("/workspace", "/workspace/ABCDFSS"):
        if p not in sys.path:
            sys.path.insert(0, p)

    from abcdfss_plus.benchmark import run_benchmark

    unit_out = f"/results/table4/{benchmark}_{shot}shot"
    existing_json = f"{unit_out}/results.json"
    results_vol.reload()
    if os.path.isfile(existing_json):
        with open(existing_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("results"):
            print(f"[skip] table4/{benchmark}_{shot}shot already completed", flush=True)
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
def run_part_detached(
    part: int,
    suim_max_episodes: int = 1650,
    seed: int = 42,
    num_epochs: int = 25,
) -> dict:
    if part == 0:
        unit_pairs = [(b, s) for b in BENCHMARKS for s in SHOTS]
        out_name = "table4_all"
    elif part in PART_UNITS:
        unit_pairs = PART_UNITS[part]
        out_name = f"table4_part{part}"
    else:
        raise ValueError(f"Invalid part={part}. Choose 1, 2, 3, or 0 (all).")

    units = [(b, s, suim_max_episodes, seed, num_epochs) for b, s in unit_pairs]
    print(f"[detached] Launching {len(units)} Table 4 (every-episode) GPU units for part={part}: {unit_pairs}", flush=True)
    rows = list(run_table4_unit.starmap(units))

    json_path = f"/results/{out_name}.json"
    csv_path = f"/results/{out_name}.csv"
    payload = {
        "method": "ABCDFSS-Plus",
        "table": "Table 4 (no-pp)",
        "part": part,
        "platform": "Modal (T4)",
        "inference_mode": "every-episode",
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "benchmark",
        "shot",
        "mIoU",
        "FB-IoU",
        "episode_count",
        "available_episodes",
        "elapsed_sec",
        "seed",
        "inference_mode",
        "num_epochs",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    results_vol.commit()
    print(f"[detached] Saved {json_path} and {csv_path}", flush=True)
    return payload


@app.local_entrypoint()
def main(
    part: int = 1,
    suim_max_episodes: int = 1650,
    seed: int = 42,
    num_epochs: int = 25,
    skip_download: bool = False,
    only_unit: str = "",
):
    if only_unit:
        b, s_str = only_unit.split(":")
        unit_pairs = [(b.strip(), int(s_str.strip()))]
    elif part in PART_UNITS:
        unit_pairs = PART_UNITS[part]
    else:
        unit_pairs = [(b, s) for b in BENCHMARKS for s in SHOTS]
    required_keys = sorted({b for b, _ in unit_pairs})

    if not skip_download:
        kaggle_env = _load_kaggle_auth()
        ensure_datasets.remote(kaggle_env, required_keys)

    if only_unit:
        b, s = unit_pairs[0]
        row = run_table4_unit.remote(b, s, suim_max_episodes, seed, num_epochs)
        print(f"Completed {b}_{s}shot: {row}", flush=True)
        return

    payload = run_part_detached.remote(
        part=part,
        suim_max_episodes=suim_max_episodes,
        seed=seed,
        num_epochs=num_epochs,
    )
    out_dir = os.path.join(ROOT_DIR, "results", "parts")
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"table4_part{part}" if part in PART_UNITS else "table4_all"
    with open(os.path.join(out_dir, f"{out_name}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Completed part={part}. Saved to results/parts/{out_name}.json", flush=True)
