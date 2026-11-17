import json
import os
import re
import subprocess
import time

import modal

APP_NAME = "abcdfss-baseline-lung"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "tensorboardX",
        "pillow",
        "scipy",
        "scikit-image",
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
)

datasets_vol = modal.Volume.from_name("abcdfss-datasets", create_if_missing=True)
results_vol = modal.Volume.from_name("abcdfss-results", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="T4",
    volumes={"/data": datasets_vol, "/results": results_vol},
    timeout=43200,
)
def run_baseline_lung(nshot: int) -> dict:
    out_dir = f"/results/baseline_table4/lung_{nshot}shot"
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/stdout.log"

    cmd = [
        "python",
        "-u",
        "main.py",
        "--benchmark",
        "lung",
        "--datapath",
        "/data/lung",
        "--nshot",
        str(nshot),
        "--adapt-to",
        "every-episode",
        "--postprocessing",
        "off",
        "--logpath",
        out_dir,
        "--verbosity",
        "1",
    ]
    print(f"[baseline lung {nshot}-shot] Running: {' '.join(cmd)}", flush=True)

    t0 = time.time()
    lines = []
    proc = subprocess.Popen(
        cmd,
        cwd="/workspace/ABCDFSS",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8") as lf:
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            lines.append(line.rstrip("\n"))
            print(f"[lung {nshot}-shot] {line}", end="", flush=True)
    rc = proc.wait()
    elapsed = round(time.time() - t0, 2)

    if rc != 0:
        results_vol.commit()
        raise RuntimeError(f"ABCDFSS/main.py failed with exit code {rc} for lung {nshot}-shot")

    full_text = "\n".join(lines)
    m = re.search(
        r"Result m\|FB:\s*\(tensor\(([0-9.]+).*?\),\s*tensor\(([0-9.]+)",
        full_text,
    )
    miou = round(float(m.group(1)), 2) if m else None
    fbiou = round(float(m.group(2)), 2) if m else None

    result = {
        "method": "ABCDFSS (original)",
        "benchmark": "lung",
        "shot": nshot,
        "adapt_to": "every-episode",
        "postprocessing": "off",
        "mIoU": miou,
        "FB-IoU": fbiou,
        "elapsed_sec": elapsed,
    }
    with open(f"{out_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"[baseline lung {nshot}-shot] Completed: {result}", flush=True)
    return result


@app.function(volumes={"/results": results_vol}, timeout=43200)
def run_all_baseline_lung() -> dict:
    rows = list(run_baseline_lung.map([1, 5]))
    summary = {
        "method": "ABCDFSS (original)",
        "table": "Table 4 (every-episode, no-pp)",
        "benchmark": "lung",
        "platform": "Modal (T4)",
        "results": rows,
    }
    os.makedirs("/results/baseline_table4", exist_ok=True)
    with open("/results/baseline_table4/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    return summary
