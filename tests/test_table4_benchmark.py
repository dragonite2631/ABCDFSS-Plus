import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABCDFSS_DIR = os.path.join(ROOT_DIR, "ABCDFSS")
for p in (ABCDFSS_DIR, ROOT_DIR):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import improved_abcdfss
import merge_table4_results
from abcdfss_plus.benchmark import run_benchmark
from abcdfss_plus.preflight import PreflightError, validate_all_datasets


def _save_rgb(path: str, size=(32, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _save_mask(path: str, size=(32, 32)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[4:20, 4:20] = 255
    Image.fromarray(arr).save(path)


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 64, kernel_size=1)
        self.channels = [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3

    def extract_feats(self, img):
        device = img.device
        feats = []
        base = self.proj(img)
        for idx, ch in enumerate(self.channels):
            spatial = 50 if idx < 3 else (25 if idx < 7 else (13 if idx < 13 else 7))
            f = torch.nn.functional.adaptive_avg_pool2d(base, (spatial, spatial))
            feats.append(f.repeat(1, ch // 64, 1, 1).to(device))
        return feats


class TestTable4Benchmark(unittest.TestCase):
    def test_compatibility_reexports(self):
        for sym in ("SpatialContextAdapter", "ImprovedDAMatComparison", "ImprovedFeatureMaker"):
            self.assertTrue(hasattr(improved_abcdfss, sym))

    def test_preflight_all_five_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            dg = os.path.join(tmp, "deepglobe")
            for c in ["1", "2", "3", "4", "5", "6"]:
                _save_rgb(os.path.join(dg, c, "test", "origin", "img1.jpg"))
                _save_mask(os.path.join(dg, c, "test", "groundtruth", "img1.png"))

            isic = os.path.join(tmp, "isic")
            for c in ["1", "2", "3"]:
                _save_rgb(os.path.join(isic, "ISIC2018_Task1-2_Training_Input", c, f"ISIC_{c}.jpg"))
                _save_mask(os.path.join(isic, "ISIC2018_Task1_Training_GroundTruth", f"ISIC_{c}_segmentation.png"))

            lung = os.path.join(tmp, "lung")
            _save_rgb(os.path.join(lung, "CXR_png", "case1.png"))
            _save_mask(os.path.join(lung, "masks", "case1_mask.png"))

            fss = os.path.join(tmp, "fss")
            with open(os.path.join(ABCDFSS_DIR, "data", "splits", "fss", "test.txt"), "r", encoding="utf-8") as f:
                fss_classes = [line.strip() for line in f if line.strip()]
            for cat in fss_classes:
                _save_rgb(os.path.join(fss, "FSS-1000", cat, "1.jpg"))
                _save_mask(os.path.join(fss, "FSS-1000", cat, "1.png"))

            suim = os.path.join(tmp, "suim", "suim_merged")
            for c in ["FV", "HD", "PF", "RI", "RO", "SR", "WR"]:
                _save_rgb(os.path.join(suim, "images", f"u_{c}.jpg"))
                _save_mask(os.path.join(suim, "masks", c, f"u_{c}.bmp"))

            roots = {"deepglobe": dg, "isic": isic, "lung": lung, "fss": fss, "suim": os.path.join(tmp, "suim")}
            res = validate_all_datasets(roots, abcdfss_dir=ABCDFSS_DIR)
            self.assertEqual(res["isic"].num_classes, 3)
            self.assertEqual(res["fss"].num_classes, len(fss_classes))

            with self.assertRaises(PreflightError):
                validate_all_datasets(dict(roots, lung=os.path.join(tmp, "missing")), abcdfss_dir=ABCDFSS_DIR, benchmarks=["lung"])

    def test_every_episode_execution_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            lung = os.path.join(tmp, "lung")
            for i in range(3):
                _save_rgb(os.path.join(lung, "CXR_png", f"p{i}.png"), size=(64, 64))
                _save_mask(os.path.join(lung, "masks", f"p{i}_mask.png"), size=(64, 64))

            out_dir = os.path.join(tmp, "out")
            res = run_benchmark(
                dataset_roots={"lung": lung},
                output_dir=out_dir,
                benchmarks=["lung"],
                shots=[1],
                max_episodes_override=1,
                seed=42,
                num_epochs=1,
                fit_every_episode=True,
                abcdfss_dir=ABCDFSS_DIR,
                backbone_factory=_DummyBackbone,
                require_cuda=False,
            )
            self.assertEqual(res["results"][0]["inference_mode"], "every-episode")
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "results.csv")))

            parts_dir = os.path.join(tmp, "parts")
            os.makedirs(parts_dir, exist_ok=True)
            with open(os.path.join(parts_dir, "table4_part1.json"), "w", encoding="utf-8") as f:
                json.dump(res, f)
            merged = merge_table4_results.load_part_rows(parts_dir)
            self.assertIn(("lung", 1), merged)
            html_row = merge_table4_results.format_table4_html_row(merged)
            self.assertIn("ABCDFSS-Plus (every-episode, no-pp)", html_row)

    def test_mid_unit_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            lung = os.path.join(tmp, "lung")
            for i in range(3):
                _save_rgb(os.path.join(lung, "CXR_png", f"p{i}.png"), size=(64, 64))
                _save_mask(os.path.join(lung, "masks", f"p{i}_mask.png"), size=(64, 64))

            out_dir = os.path.join(tmp, "out")
            ckpt_path = os.path.join(out_dir, "lung_1shot_ckpt.pt")

            def _interrupt_on_first_ckpt():
                raise RuntimeError("simulated preemption")

            with self.assertRaisesRegex(RuntimeError, "simulated preemption"):
                run_benchmark(
                    dataset_roots={"lung": lung},
                    output_dir=out_dir,
                    benchmarks=["lung"],
                    shots=[1],
                    max_episodes_override=3,
                    seed=42,
                    num_epochs=1,
                    fit_every_episode=True,
                    abcdfss_dir=ABCDFSS_DIR,
                    backbone_factory=_DummyBackbone,
                    require_cuda=False,
                    checkpoint_every=1,
                    on_checkpoint=_interrupt_on_first_ckpt,
                )

            self.assertTrue(os.path.isfile(ckpt_path))

            ckpt_calls = []
            res = run_benchmark(
                dataset_roots={"lung": lung},
                output_dir=out_dir,
                benchmarks=["lung"],
                shots=[1],
                max_episodes_override=3,
                seed=42,
                num_epochs=1,
                fit_every_episode=True,
                abcdfss_dir=ABCDFSS_DIR,
                backbone_factory=_DummyBackbone,
                require_cuda=False,
                checkpoint_every=1,
                on_checkpoint=lambda: ckpt_calls.append(1),
            )

            self.assertFalse(os.path.exists(ckpt_path))
            self.assertEqual(len(ckpt_calls), 1)
            self.assertEqual(res["results"][0]["episode_count"], 3)
            self.assertEqual(res["results"][0]["inference_mode"], "every-episode")
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "results.csv")))

    def test_dwla_multishot_mask_alignment(self):
        # Construct k=2 supports where shot 0 is all FG ([1, 0]) and shot 1 is all BG ([-1, 0]).
        # With h=2, w=2:
        # - In (k, h, w) order: first 4 pixels are FG, last 4 pixels are BG.
        # - In horizontal smask_row [1, h, k*w] = [1, 2, 4]: row 0 has 2 FG then 2 BG,
        #   so naive .reshape(1, k*h*w) would misalign pixels (0,1,0,0) with FG features!
        sft = torch.zeros(1, 2, 2, 2, 2)
        sft[0, 0, 0, :, :] = 1.0   # shot 0: vector [1, 0] everywhere (FG)
        sft[0, 1, 0, :, :] = -1.0  # shot 1: vector [-1, 0] everywhere (BG)

        m0 = torch.ones(1, 2, 2)
        m1 = torch.zeros(1, 2, 2)
        smask_row = torch.cat([m0, m1], dim=-1)       # [1, 2, 4]
        smask_stacked = torch.stack([m0, m1], dim=1)  # [1, 2, 2, 2]

        delta_from_row = improved_abcdfss.compute_layer_discriminability(sft, smask_row)
        delta_from_stacked = improved_abcdfss.compute_layer_discriminability(sft, smask_stacked)

        # True fg_mean = [1, 0], bg_mean = [-1, 0] => sim_fg_fg=1.0, sim_fg_bg=-1.0 => delta=2.0
        self.assertAlmostEqual(delta_from_row, 2.0, places=5)
        self.assertAlmostEqual(delta_from_stacked, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
