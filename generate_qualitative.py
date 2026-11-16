import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ABCDFSS_DIR = os.path.join(ROOT_DIR, "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

from core.backbone import Backbone
from utils import commonutils as utils
from utils import segutils
from improved_abcdfss import ImprovedFeatureMaker, ImprovedDAMatComparison
from run_experiments import make_config, build_fixed_episodes


def denorm_img(t):
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return img


def overlay(img, mask, color=(1.0, 0.2, 0.2), alpha=0.45):
    out = img.copy()
    fg = mask > 0.5
    out[fg] = (1 - alpha) * img[fg] + alpha * np.array(color)
    return np.clip(out, 0, 1)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = Backbone("resnet50").to(device)
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

    samples = []
    # 2 Lung examples + 2 SUIM examples
    _, lung_eps = build_fixed_episodes("lung", os.path.join(ROOT_DIR, "datasets", "lung"), nshot=1, max_episodes=5, seed=42)
    samples.append(("Lung CXR #1", lung_eps[0], range(0, 1)))
    samples.append(("Lung CXR #2", lung_eps[2], range(0, 1)))

    _, suim_eps = build_fixed_episodes("suim", os.path.join(ROOT_DIR, "datasets", "suim"), nshot=1, max_episodes=14, seed=42)
    samples.append(("SUIM (Diver)", suim_eps[1], range(0, 7)))
    samples.append(("SUIM (Robot)", suim_eps[4], range(0, 7)))

    fig, axes = plt.subplots(len(samples), 4, figsize=(14, 3.5 * len(samples)))
    col_titles = ["Support + Mask", "Query + GT", "Baseline (ABCDFSS)", "Improved (ABCDFSS-Plus)"]

    for r, (label, batch, cids) in enumerate(samples):
        b = utils.to_cuda(batch) if device.type == "cuda" else batch
        q_img, s_img, s_mask, cid = b["query_img"], b["support_imgs"], b["support_masks"], b["class_id"].item()
        q_gt = b["query_mask"][0].cpu().numpy()

        # Baseline
        utils.fix_randseed(2)
        cfg_b = make_config(fit_every_episode=True)
        fm_b = ImprovedFeatureMaker(backbone.extract_feats, cids, cfg_b)
        da_b = ImprovedDAMatComparison()
        qf_b, sf_b = fm_b.taskAdapt(q_img, s_img, s_mask, cid)
        lm_b = da_b.forward(qf_b, sf_b, s_mask)
        th_b = segutils.calcthresh(lm_b, s_mask, "pred_mean")
        pred_b = (lm_b > th_b).float()[0].cpu().numpy()

        # Improved
        utils.fix_randseed(2)
        cfg_i = make_config(fit_every_episode=True)
        fm_i = ImprovedFeatureMaker(backbone.extract_feats, cids, cfg_i, use_dwconv_adapter=True, use_self_rec=True)
        da_i = ImprovedDAMatComparison(use_dwla=True, use_cosine_affinity=True)
        qf_i, sf_i = fm_i.taskAdapt(q_img, s_img, s_mask, cid)
        lm_i = da_i.forward(qf_i, sf_i, s_mask)
        th_i = segutils.calcthresh(lm_i, s_mask, "pred_mean")
        pred_i = (lm_i > th_i).float()[0].cpu().numpy()

        def iou(p, g):
            return (np.logical_and(p > 0.5, g > 0.5).sum() / (np.logical_or(p > 0.5, g > 0.5).sum() + 1e-8)) * 100

        s_np = denorm_img(s_img[0, 0])
        sm_np = s_mask[0, 0].cpu().numpy()
        q_np = denorm_img(q_img[0])

        imgs = [
            overlay(s_np, sm_np, color=(0.1, 0.85, 0.4)),
            overlay(q_np, q_gt, color=(0.2, 0.5, 1.0)),
            overlay(q_np, pred_b, color=(0.95, 0.25, 0.25)),
            overlay(q_np, pred_i, color=(0.95, 0.25, 0.25)),
        ]
        subtitles = [
            label,
            "Ground Truth",
            f"IoU: {iou(pred_b, q_gt):.1f}%",
            f"IoU: {iou(pred_i, q_gt):.1f}%",
        ]
        for c in range(4):
            axes[r, c].imshow(imgs[c])
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=11, fontweight="bold")
            axes[r, c].text(
                5, 22, subtitles[c], color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=2)
            )

    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "results", "qualitative_comparison.png"), dpi=150)
    plt.close()
    print("Saved results/qualitative_comparison.png")


if __name__ == "__main__":
    main()
