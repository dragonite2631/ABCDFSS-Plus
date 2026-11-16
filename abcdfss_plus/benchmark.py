import csv
import json
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABCDFSS_DIR = os.path.join(ROOT_DIR, "ABCDFSS")
for p in (ROOT_DIR, ABCDFSS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.backbone import Backbone
from data.dataset import FSSDataset
from eval.evaluation import Evaluator
try:
    from eval.logger import AverageMeter
except ModuleNotFoundError:
    sys.modules.setdefault("tensorboardX", types.SimpleNamespace(SummaryWriter=object))
    from eval.logger import AverageMeter
from utils import commonutils as utils
from utils import segutils

from abcdfss_plus import (
    ImprovedDAMatComparison,
    ImprovedFeatureMaker,
    make_plus_config,
)
from abcdfss_plus.preflight import DatasetPreflightResult, validate_all_datasets

DEFAULT_BENCHMARKS: List[str] = ["deepglobe", "isic", "lung", "fss", "suim"]
DEFAULT_SHOTS: List[int] = [1, 5]


@contextmanager
def _working_directory(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _close_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        finally:
            logger.removeHandler(h)


def _setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("abcdfss_plus_table4")
    logger.setLevel(logging.INFO)
    _close_logger(logger)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(output_dir, "benchmark.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _unit_key(benchmark: str, shot: int) -> str:
    return f"{benchmark}_{shot}shot"


def _load_progress(progress_path: str) -> dict:
    if os.path.isfile(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_units": {}, "failed_unit": None}


def _save_json(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _write_outputs(output_dir: str, config_payload: dict, progress: dict) -> None:
    completed = list(progress["completed_units"].values())
    completed.sort(
        key=lambda r: (
            DEFAULT_BENCHMARKS.index(r["benchmark"]) if r["benchmark"] in DEFAULT_BENCHMARKS else 999,
            r["shot"],
        )
    )

    results_payload = {
        "method": "ABCDFSS-Plus",
        "table": "Table 4 (every-episode, no-pp)",
        "config": config_payload,
        "results": completed,
    }
    _save_json(os.path.join(output_dir, "results.json"), results_payload)

    csv_path = os.path.join(output_dir, "results.csv")
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
    tmp_csv = csv_path + ".tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in completed:
            writer.writerow({k: row.get(k) for k in fieldnames})
    os.replace(tmp_csv, csv_path)


def evaluate_single_unit(
    benchmark: str,
    shot: int,
    dataset_root: str,
    backbone,
    device: torch.device,
    seed: int = 42,
    num_epochs: int = 25,
    max_episodes: Optional[int] = None,
    fit_every_episode: bool = True,
    abcdfss_dir: str = ABCDFSS_DIR,
    logger: Optional[logging.Logger] = None,
    log_every: int = 100,
) -> dict:
    utils.fix_randseed(seed)
    with _working_directory(abcdfss_dir):
        FSSDataset.initialize(img_size=400, datapath=dataset_root)
        base_loader = FSSDataset.build_dataloader(
            benchmark, bsz=1, nworker=0, fold=0, split="test", shot=shot
        )
        dataset = base_loader.dataset

    effective_workers = 4 if (os.name != "nt" and device.type == "cuda") else 0
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=(device.type == "cuda"),
    )

    available_episodes = len(dataset)
    target_episodes = (
        min(available_episodes, max_episodes)
        if max_episodes is not None and max_episodes > 0
        else available_episodes
    )

    config = make_plus_config(
        fit_every_episode=fit_every_episode,
        num_epochs=num_epochs,
    )
    feat_maker = ImprovedFeatureMaker(
        backbone.extract_feats,
        dataset.class_ids,
        config,
        use_dwconv_adapter=True,
        use_self_rec=True,
    )
    damat_comp = ImprovedDAMatComparison(
        use_dwla=True,
        use_cosine_affinity=True,
    )
    Evaluator.initialize()
    meter = AverageMeter(dataset, device)

    t0 = time.time()
    processed = 0
    for idx, batch in enumerate(dataloader):
        if idx >= target_episodes:
            break
        b = utils.to_cuda(batch) if device.type == "cuda" else batch
        q_img = b["query_img"]
        s_img = b["support_imgs"]
        s_mask = b["support_masks"]
        class_id = b["class_id"].item()

        q_feat_t, s_feat_t = feat_maker.taskAdapt(q_img, s_img, s_mask, class_id)
        logit_mask = damat_comp.forward(q_feat_t, s_feat_t, s_mask)
        thresh = segutils.calcthresh(logit_mask, s_mask, "pred_mean")
        pred_mask = (logit_mask > thresh).float()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, b)
        meter.update(area_inter, area_union, torch.tensor(class_id).to(device), loss=None)
        processed += 1

        if logger and log_every > 0 and processed % log_every == 0:
            cur_miou, cur_fbiou = meter.compute_iou()
            logger.info(
                f"[{benchmark} {shot}-shot] {processed}/{target_episodes} episodes | "
                f"running mIoU={cur_miou.item():.2f}% FB-IoU={cur_fbiou.item():.2f}% "
                f"elapsed={time.time() - t0:.1f}s"
            )

    elapsed = time.time() - t0
    miou, fb_iou = meter.compute_iou()
    return {
        "benchmark": benchmark,
        "shot": shot,
        "mIoU": round(float(miou.item()), 2),
        "FB-IoU": round(float(fb_iou.item()), 2),
        "episode_count": processed,
        "available_episodes": available_episodes,
        "elapsed_sec": round(elapsed, 2),
        "seed": seed,
        "inference_mode": "every-episode" if fit_every_episode else "quick-infer",
        "num_epochs": num_epochs,
        "resolved_root": dataset_root,
    }


def run_benchmark(
    dataset_roots: Dict[str, str],
    output_dir: str,
    benchmarks: Optional[List[str]] = None,
    shots: Optional[List[int]] = None,
    suim_max_episodes: Optional[int] = 1650,
    max_episodes_override: Optional[int] = None,
    seed: int = 42,
    num_epochs: int = 25,
    fit_every_episode: bool = True,
    abcdfss_dir: str = ABCDFSS_DIR,
    backbone_factory: Optional[Callable[[], torch.nn.Module]] = None,
    require_cuda: bool = False,
) -> dict:
    logger = _setup_logger(output_dir)
    target_benchmarks = benchmarks or list(DEFAULT_BENCHMARKS)
    target_shots = shots or list(DEFAULT_SHOTS)

    try:
        logger.info("Running dataset preflight checks...")
        preflight: Dict[str, DatasetPreflightResult] = validate_all_datasets(
            dataset_roots, abcdfss_dir=abcdfss_dir, benchmarks=target_benchmarks
        )
        for bname, res in preflight.items():
            logger.info(f"  [OK] {bname}: {res.marker_summary} (root={res.resolved_root})")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif require_cuda:
            raise RuntimeError("CUDA is required for this benchmark run, but no GPU was detected.")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        try:
            backbone = backbone_factory() if backbone_factory is not None else Backbone("resnet50")
            backbone = backbone.to(device)
            backbone.eval()
        except Exception as exc:
            raise RuntimeError("Failed to initialize ResNet-50 pretrained backbone.") from exc

        config_payload = {
            "method": "ABCDFSS-Plus",
            "table": "Table 4 (no-pp)",
            "benchmarks": target_benchmarks,
            "shots": target_shots,
            "suim_max_episodes": suim_max_episodes,
            "max_episodes_override": max_episodes_override,
            "seed": seed,
            "num_epochs": num_epochs,
            "inference_mode": "every-episode" if fit_every_episode else "quick-infer",
            "device": str(device),
            "improvements": {
                "use_dwla": True,
                "use_dwconv_adapter": True,
                "use_self_rec": True,
                "use_cosine_affinity": True,
            },
            "resolved_roots": {k: v.resolved_root for k, v in preflight.items()},
        }
        _save_json(os.path.join(output_dir, "config.json"), config_payload)

        progress_path = os.path.join(output_dir, "progress.json")
        progress = _load_progress(progress_path)

        for bname in target_benchmarks:
            resolved_root = preflight[bname].resolved_root
            for shot in target_shots:
                key = _unit_key(bname, shot)
                if key in progress["completed_units"]:
                    logger.info(f"Skipping completed unit: {key}")
                    continue

                unit_cap = max_episodes_override
                if unit_cap is None and bname == "suim" and suim_max_episodes and suim_max_episodes > 0:
                    unit_cap = suim_max_episodes

                logger.info(
                    f"Starting unit {key} (cap={unit_cap if unit_cap is not None else 'full'})..."
                )
                try:
                    unit_result = evaluate_single_unit(
                        benchmark=bname,
                        shot=shot,
                        dataset_root=resolved_root,
                        backbone=backbone,
                        device=device,
                        seed=seed,
                        num_epochs=num_epochs,
                        max_episodes=unit_cap,
                        fit_every_episode=fit_every_episode,
                        abcdfss_dir=abcdfss_dir,
                        logger=logger,
                    )
                except Exception as exc:
                    progress["failed_unit"] = {"unit": key, "error": str(exc)}
                    _save_json(progress_path, progress)
                    raise

                progress["completed_units"][key] = unit_result
                progress["failed_unit"] = None
                _save_json(progress_path, progress)
                _write_outputs(output_dir, config_payload, progress)
                logger.info(
                    f"Completed {key}: mIoU={unit_result['mIoU']}% | "
                    f"FB-IoU={unit_result['FB-IoU']}% | "
                    f"episodes={unit_result['episode_count']} | "
                    f"time={unit_result['elapsed_sec']}s"
                )

        _write_outputs(output_dir, config_payload, progress)
        logger.info(f"All requested units finished. Outputs saved in {output_dir}")
        with open(os.path.join(output_dir, "results.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        _close_logger(logger)
