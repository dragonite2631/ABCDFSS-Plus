import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ABCDFSS_DIR = os.path.join(ROOT_DIR, "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

from data.dataset import FSSDataset
from core.backbone import Backbone
from eval.evaluation import Evaluator
from eval.logger import AverageMeter
from utils import commonutils as utils
from utils import segutils
import core.contrastivehead as ctrutils

from improved_abcdfss import (
    ImprovedFeatureMaker,
    ImprovedDAMatComparison,
    compute_layer_discriminability,
)


def make_config(
    fit_every_episode=False,
    q_nceloss=True,
    s_nceloss=True,
    keepvarloss=True,
    o_t_contr_proto_loss=True,
    num_epochs=25,
    lr=1e-2,
):
    config = ctrutils.ContrastiveConfig()
    config.fitting.protoloss = False
    config.fitting.o_t_contr_proto_loss = o_t_contr_proto_loss
    config.fitting.selfattentionloss = False
    config.fitting.keepvarloss = keepvarloss
    config.fitting.symmetricloss = False
    config.fitting.q_nceloss = q_nceloss
    config.fitting.s_nceloss = s_nceloss
    config.fitting.num_epochs = num_epochs
    config.fitting.lr = lr
    config.fitting.debug = False
    config.model.out_channels = 64
    config.model.debug = False
    config.featext.fit_every_episode = fit_every_episode
    config.aug.blurkernelsize = [1]
    config.aug.n_transformed_imgs = 2
    config.aug.maxjitter = 0.0
    config.aug.maxangle = 0
    config.aug.maxscale = 1
    config.aug.maxshear = 20
    config.aug.apply_affine = True
    config.aug.debug = False
    return config


def build_fixed_episodes(benchmark, datapath, nshot=1, max_episodes=None, seed=42):
    utils.fix_randseed(seed)
    FSSDataset.initialize(img_size=400, datapath=datapath)
    dataloader = FSSDataset.build_dataloader(benchmark, bsz=1, nworker=0, fold=0, split="test", shot=nshot)
    episodes = []
    for idx, batch in enumerate(dataloader):
        episodes.append(batch)
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    return dataloader.dataset, episodes


def compute_embedding_deltas(backbone, feat_maker, episodes, device, max_samples=30):
    """
    Reproduces Table 7/8 of the paper:
    Measures Intra-Support and Inter-Query-Support cosine similarity delta
    for Low (layers 3..6), Mid (layers 7..12), High (layers 13..15) before and after TA.
    """
    layer_groups = {
        "L": list(range(3, 7)),
        "M": list(range(7, 13)),
        "H": list(range(13, 16)),
    }
    stats = {
        stage: {g: {"intra": [], "inter": []} for g in layer_groups}
        for stage in ["before_ta", "after_ta"]
    }

    for batch in episodes[:max_samples]:
        q_img = batch["query_img"].to(device)
        s_img = batch["support_imgs"].to(device)
        s_mask = batch["support_masks"].to(device)
        q_mask = batch["query_mask"].to(device)
        class_id = batch["class_id"].item()

        with torch.no_grad():
            q_bb = backbone.extract_feats(q_img)
            bsz, k, c, h, w = s_img.shape
            s_bb = backbone.extract_feats(s_img.view(-1, c, h, w))
            s_bb = [l.view(bsz, k, *l.shape[1:]) for l in s_bb]

        q_ta, s_ta = feat_maker.taskAdapt(q_img, s_img, s_mask, class_id)

        for stage, (qf_list, sf_list) in [("before_ta", (q_bb, s_bb)), ("after_ta", (q_ta, s_ta))]:
            for gname, l_indices in layer_groups.items():
                intra_vals, inter_vals = [], []
                for l in l_indices:
                    if l >= len(qf_list) or qf_list[l] is None:
                        continue
                    qf = qf_list[l].detach()
                    sf = sf_list[l].detach()
                    hs, ws = sf.shape[-2:]
                    sm = torch.cat([segutils.downsample_mask(m, hs, ws) for m in s_mask.unbind(1)], -1)
                    intra_vals.append(compute_layer_discriminability(sf, sm))

                    # Inter query-support delta
                    hq, wq = qf.shape[-2:]
                    qm = segutils.downsample_mask(q_mask, hq, wq).reshape(-1)
                    sm_flat = sm.reshape(-1)
                    qf_flat = F.normalize(qf.permute(0, 2, 3, 1).reshape(-1, qf.shape[1]), p=2, dim=-1)
                    sf_flat = F.normalize(sf.permute(0, 1, 3, 4, 2).reshape(-1, sf.shape[2]), p=2, dim=-1)

                    q_fg = qf_flat[qm > 0.5]
                    s_fg = sf_flat[sm_flat > 0.5]
                    s_bg = sf_flat[sm_flat <= 0.5]
                    if len(q_fg) > 0 and len(s_fg) > 0 and len(s_bg) > 0:
                        q_fg_m = q_fg.mean(0)
                        s_fg_m = s_fg.mean(0)
                        s_bg_m = s_bg.mean(0)
                        inter_vals.append((torch.dot(q_fg_m, s_fg_m) - torch.dot(q_fg_m, s_bg_m)).item())
                if intra_vals:
                    stats[stage][gname]["intra"].append(float(np.mean(intra_vals)))
                if inter_vals:
                    stats[stage][gname]["inter"].append(float(np.mean(inter_vals)))

    summary = {}
    for stage in stats:
        summary[stage] = {}
        for g in layer_groups:
            summary[stage][g] = {
                "intra_delta": round(float(np.mean(stats[stage][g]["intra"])) * 100, 2) if stats[stage][g]["intra"] else 0.0,
                "inter_delta": round(float(np.mean(stats[stage][g]["inter"])) * 100, 2) if stats[stage][g]["inter"] else 0.0,
            }
    return summary


def evaluate_variant(
    name,
    dataset,
    episodes,
    backbone,
    device,
    use_ta=True,
    q_nceloss=True,
    s_nceloss=True,
    keepvarloss=True,
    o_t_contr_proto_loss=True,
    fit_every_episode=False,
    use_dwla=False,
    use_dwconv_adapter=False,
    use_self_rec=False,
    use_cosine_affinity=False,
    seed=2,
):
    utils.fix_randseed(seed)
    config = make_config(
        fit_every_episode=fit_every_episode,
        q_nceloss=q_nceloss,
        s_nceloss=s_nceloss,
        keepvarloss=keepvarloss,
        o_t_contr_proto_loss=o_t_contr_proto_loss,
    )
    feat_maker = ImprovedFeatureMaker(
        backbone.extract_feats,
        dataset.class_ids,
        config,
        use_dwconv_adapter=use_dwconv_adapter,
        use_self_rec=use_self_rec,
    )
    damat_comp = ImprovedDAMatComparison(
        use_dwla=use_dwla,
        use_cosine_affinity=use_cosine_affinity,
    )
    Evaluator.initialize()
    meter = AverageMeter(dataset, device)

    t0 = time.time()
    sample_preds = []
    last_weights = None

    for idx, batch in enumerate(episodes):
        b = utils.to_cuda(batch) if device.type == "cuda" else batch
        q_img, s_img, s_mask, class_id = (
            b["query_img"],
            b["support_imgs"],
            b["support_masks"],
            b["class_id"].item(),
        )

        if use_ta:
            q_feat_t, s_feat_t = feat_maker.taskAdapt(q_img, s_img, s_mask, class_id)
        else:
            with torch.no_grad():
                q_bb = feat_maker.extract_bb_feats(q_img)
                bsz, k, c, h, w = s_img.shape
                s_bb = feat_maker.extract_bb_feats(s_img.view(-1, c, h, w))
                s_bb = [l.view(bsz, k, *l.shape[1:]) for l in s_bb]
                q_feat_t = [None if i < 3 else l for i, l in enumerate(q_bb)]
                s_feat_t = [None if i < 3 else l for i, l in enumerate(s_bb)]

        logit_mask = damat_comp.forward(q_feat_t, s_feat_t, s_mask)
        thresh = segutils.calcthresh(logit_mask, s_mask, "pred_mean")
        pred_mask = (logit_mask > thresh).float()

        area_inter, area_union = Evaluator.classify_prediction(pred_mask, b)
        meter.update(area_inter, area_union, torch.tensor(class_id).to(device), loss=None)

        if idx < 4:
            sample_preds.append(
                {
                    "idx": idx,
                    "pred": pred_mask[0].detach().cpu().numpy(),
                    "logit": logit_mask[0].detach().cpu().numpy(),
                    "weights": damat_comp.last_layer_weights,
                }
            )
        if last_weights is None:
            last_weights = damat_comp.last_layer_weights

    elapsed = time.time() - t0
    miou, fb_iou = meter.compute_iou()
    return {
        "name": name,
        "mIoU": round(miou.item(), 2),
        "FB-IoU": round(fb_iou.item(), 2),
        "time_sec": round(elapsed, 2),
        "layer_weights": [round(w, 4) for w in last_weights] if last_weights else None,
        "sample_preds": sample_preds,
    }, feat_maker


def generate_plots(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Plot 1: Bar chart comparing methods across datasets
    fig, axes = plt.subplots(1, len(all_results["benchmarks"]), figsize=(7 * len(all_results["benchmarks"]), 5))
    if len(all_results["benchmarks"]) == 1:
        axes = [axes]

    for ax, (bname, bdata) in zip(axes, all_results["benchmarks"].items()):
        variants = bdata["variants"]
        names = [v["name"] for v in variants]
        mious = [v["mIoU"] for v in variants]
        fbious = [v["FB-IoU"] for v in variants]

        x = np.arange(len(names))
        width = 0.35
        b1 = ax.bar(x - width / 2, mious, width, label="mIoU", color="#3b82f6")
        b2 = ax.bar(x + width / 2, fbious, width, label="FB-IoU", color="#10b981")

        ax.set_title(f"{bname}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend()

        for bar in b1:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
        for bar in b2:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_charts.png"), dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lung-episodes", type=int, default=138)
    parser.add_argument("--suim-episodes", type=int, default=210)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    backbone = Backbone("resnet50").to(device)

    out_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(out_dir, "metrics.json")

    if args.resume and os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {"device": str(device), "benchmarks": {}, "embedding_deltas": {}}

    benchmarks_cfg = [
        ("Lung (1-shot)", "lung", os.path.join(ROOT_DIR, "datasets", "lung"), 1, args.lung_episodes),
        ("Lung (5-shot)", "lung", os.path.join(ROOT_DIR, "datasets", "lung"), 5, args.lung_episodes),
        ("SUIM (1-shot)", "suim", os.path.join(ROOT_DIR, "datasets", "suim"), 1, args.suim_episodes),
        ("SUIM (5-shot)", "suim", os.path.join(ROOT_DIR, "datasets", "suim"), 5, args.suim_episodes),
    ]

    variant_configs = [
        ("w/o TA", dict(use_ta=False)),
        ("w/o L_q, L_s", dict(use_ta=True, q_nceloss=False, s_nceloss=False, keepvarloss=False)),
        ("w/o L_p", dict(use_ta=True, o_t_contr_proto_loss=False)),
        ("Baseline (ABCDFSS)", dict(use_ta=True)),
        ("+ Imp1: DWLA", dict(use_ta=True, use_dwla=True)),
        ("+ Imp2: DWConv3x3", dict(use_ta=True, use_dwconv_adapter=True)),
        ("+ Imp3: SelfRec+CosAff", dict(use_ta=True, use_self_rec=True, use_cosine_affinity=True)),
        ("Full Improved", dict(use_ta=True, use_dwla=True, use_dwconv_adapter=True, use_self_rec=True, use_cosine_affinity=True)),
    ]

    for btitle, bkey, dpath, nshot, max_ep in benchmarks_cfg:
        if args.resume and btitle in summary["benchmarks"]:
            print(f"\n=== Skipping {btitle} (already in metrics.json) ===")
            continue
        print(f"\n=== Running {btitle} ({max_ep} episodes) ===")
        dataset, episodes = build_fixed_episodes(bkey, dpath, nshot=nshot, max_episodes=max_ep, seed=42)
        b_results = []
        base_fm = None
        full_fm = None

        for vname, vkwargs in variant_configs:
            res, fm = evaluate_variant(vname, dataset, episodes, backbone, device, **vkwargs)
            res_clean = {k: v for k, v in res.items() if k != "sample_preds"}
            b_results.append(res_clean)
            print(f"  [{vname:22s}] mIoU: {res['mIoU']:5.2f}% | FB-IoU: {res['FB-IoU']:5.2f}% | Time: {res['time_sec']:5.2f}s")
            if nshot == 1 and vname == "Baseline (ABCDFSS)":
                base_fm = fm
            elif nshot == 1 and vname == "Full Improved":
                full_fm = fm
            else:
                del fm
            if device.type == "cuda":
                torch.cuda.empty_cache()

        summary["benchmarks"][btitle] = {"n_episodes": len(episodes), "variants": b_results}

        # Compute embedding space discriminability delta for 1-shot benchmarks
        if nshot == 1 and base_fm is not None:
            deltas_base = compute_embedding_deltas(backbone, base_fm, episodes, device, max_samples=25)
            deltas_imp = compute_embedding_deltas(backbone, full_fm, episodes, device, max_samples=25)
            summary["embedding_deltas"][btitle] = {
                "baseline": deltas_base,
                "improved": deltas_imp,
            }
            del base_fm, full_fm
            if device.type == "cuda":
                torch.cuda.empty_cache()

        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        generate_plots(summary, out_dir)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    generate_plots(summary, out_dir)
    print("\nAll experiments completed! Results saved to results/metrics.json and results/comparison_charts.png")


if __name__ == "__main__":
    main()
