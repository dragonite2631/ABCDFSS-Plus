import os
import sys
import glob
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ABCDFSS_DIR = os.path.join(ROOT_DIR, "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

from core.backbone import Backbone
from utils import commonutils as utils
from utils import segutils
from improved_abcdfss import ImprovedFeatureMaker, ImprovedDAMatComparison
from run_experiments import make_config

st.set_page_config(
    page_title="ABCDFSS-Plus: Cross-Domain Few-Shot Segmentation",
    page_icon="🔬",
    layout="wide",
)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

SUIM_CLASSES = {
    "FV": "Fish & Vertebrates (FV)",
    "HD": "Human Divers (HD)",
    "PF": "Aquatic Plants (PF)",
    "RI": "Reefs & Invertebrates (RI)",
    "RO": "Robots & Instruments (RO)",
    "SR": "Sea-floor & Rocks (SR)",
    "WR": "Wrecks & Ruins (WR)",
}


@st.cache_resource
def load_backbone():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bb = Backbone("resnet50").to(device)
    bb.eval()
    return bb, device


def load_mask_tensor(mask_pil, size=(400, 400)):
    arr = np.array(mask_pil.convert("L"))
    t = torch.tensor(arr, dtype=torch.float32)
    t = (t >= 128).float()
    t = F.interpolate(t.unsqueeze(0).unsqueeze(0), size=size, mode="nearest").squeeze(0).squeeze(0)
    return t


def overlay_mask_np(img_pil, mask_np, color=(0, 220, 120), alpha=0.45):
    img_arr = np.array(img_pil.resize((400, 400)).convert("RGB"), dtype=np.float32)
    overlay = img_arr.copy()
    fg = mask_np > 0.5
    overlay[fg] = (1 - alpha) * img_arr[fg] + alpha * np.array(color, dtype=np.float32)
    return overlay.astype(np.uint8)


def heatmap_rgb(logit_np):
    normed = (logit_np - logit_np.min()) / (logit_np.max() - logit_np.min() + 1e-8)
    cmap = plt.get_cmap("jet")
    rgb = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def run_single_inference(
    backbone,
    device,
    q_pil,
    s_pils,
    s_mask_pils,
    num_epochs=25,
    use_dwla=False,
    dwla_temp=0.1,
    use_dwconv=False,
    use_self_rec=False,
    use_cos_aff=False,
    aff_temp=0.15,
    seed=2,
):
    utils.fix_randseed(seed)
    q_img = IMG_TRANSFORM(q_pil.convert("RGB")).unsqueeze(0).to(device)
    s_imgs = torch.stack([IMG_TRANSFORM(im.convert("RGB")) for im in s_pils]).unsqueeze(0).to(device)
    s_masks = torch.stack([load_mask_tensor(m) for m in s_mask_pils]).unsqueeze(0).to(device)

    config = make_config(fit_every_episode=True, num_epochs=num_epochs)
    feat_maker = ImprovedFeatureMaker(
        backbone.extract_feats,
        [0],
        config,
        use_dwconv_adapter=use_dwconv,
        use_self_rec=use_self_rec,
    )
    damat_comp = ImprovedDAMatComparison(
        use_dwla=use_dwla,
        dwla_temp=dwla_temp,
        use_cosine_affinity=use_cos_aff,
        aff_temp=aff_temp,
    )

    t0 = time.time()
    q_feat_t, s_feat_t = feat_maker.taskAdapt(q_img, s_imgs, s_masks, 0)
    logit_mask = damat_comp.forward(q_feat_t, s_feat_t, s_masks)
    thresh = segutils.calcthresh(logit_mask, s_masks, "pred_mean")
    pred_mask = (logit_mask > thresh).float()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000.0

    return {
        "logit": logit_mask[0].detach().cpu().numpy(),
        "pred": pred_mask[0].detach().cpu().numpy(),
        "thresh": float(thresh.item()),
        "fg_ratio": float(pred_mask.mean().item() * 100.0),
        "time_ms": elapsed_ms,
        "layer_weights": damat_comp.last_layer_weights,
        "layer_deltas": damat_comp.last_layer_deltas,
    }


def compute_iou_np(pred, gt):
    inter = np.logical_and(pred > 0.5, gt > 0.5).sum()
    union = np.logical_or(pred > 0.5, gt > 0.5).sum()
    return (inter / (union + 1e-8)) * 100.0


def get_dataset_samples(dataset_name, suim_cat="RO"):
    if dataset_name == "Chest X-ray (Lung)":
        img_dir = os.path.join(ROOT_DIR, "datasets", "lung", "CXR_png")
        mask_dir = os.path.join(ROOT_DIR, "datasets", "lung", "masks")
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        pairs = []
        for mp in masks:
            stem = os.path.basename(mp)[:-9]
            ip = os.path.join(img_dir, stem + ".png")
            if os.path.exists(ip):
                pairs.append((ip, mp, stem))
        return pairs
    else:
        img_dir = os.path.join(ROOT_DIR, "datasets", "suim", "images")
        mask_dir = os.path.join(ROOT_DIR, "datasets", "suim", "masks", suim_cat)
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        pairs = []
        for mp in masks:
            stem = os.path.basename(mp)[:-4]
            ip = os.path.join(img_dir, stem + ".jpg")
            if os.path.exists(ip):
                pairs.append((ip, mp, stem))
        return pairs


def main():
    st.title("🔬 ABCDFSS-Plus: Cross-Domain Few-Shot Segmentation Demo")
    st.caption(
        "Reproduction & Improvements of *Adapt Before Comparison: A New Perspective on Cross-Domain Few-Shot Segmentation* (CVPR 2024)"
    )

    backbone, device = load_backbone()

    tab_demo, tab_bench = st.tabs(["🎯 Interactive Segmentation Demo", "📊 Benchmark & Ablation Results"])

    with st.sidebar:
        st.header("⚙️ Task Configuration")
        st.info(f"Compute Device: **{str(device).upper()}**")

        input_mode = st.radio("Input Source", ["Dataset Sample", "Upload Custom Images"])
        nshot = st.slider("Number of Support Shots (k)", min_value=1, max_value=5, value=1)

        st.divider()
        st.header("🚀 Improved Method Settings")
        num_epochs = st.slider("Test-Time Adaptation Epochs", 5, 40, 25, step=5)
        use_dwla = st.checkbox("Imp 1: DWLA (Discriminability-Weighted Layers)", value=True)
        dwla_temp = st.slider("DWLA Temperature (τ_w)", 0.02, 0.50, 0.10, step=0.02)
        use_dwconv = st.checkbox("Imp 2: Residual DWConv3x3 Spatial Adapter", value=True)
        use_self_rec = st.checkbox("Imp 3a: Support Self-Reconstruction Loss", value=True)
        use_cos_aff = st.checkbox("Imp 3b: Cosine-Temp Dense Affinity", value=True)
        aff_temp = st.slider("Affinity Temperature (τ_aff)", 0.05, 0.50, 0.15, step=0.05)

    with tab_demo:
        q_pil, q_gt_pil = None, None
        s_pils, s_mask_pils = [], []

        if input_mode == "Dataset Sample":
            c1, c2 = st.columns([1, 2])
            with c1:
                ds_choice = st.selectbox("Target Domain Dataset", ["Chest X-ray (Lung)", "SUIM (Underwater)"])
            with c2:
                suim_cat = "RO"
                if ds_choice == "SUIM (Underwater)":
                    suim_cat = st.selectbox(
                        "SUIM Semantic Class",
                        list(SUIM_CLASSES.keys()),
                        format_func=lambda k: SUIM_CLASSES[k],
                    )

            pairs = get_dataset_samples(ds_choice, suim_cat)
            if len(pairs) < nshot + 1:
                st.error("Dataset files not found or insufficient samples.")
                return

            names = [p[2] for p in pairs]
            col_q, col_s = st.columns(2)
            with col_q:
                q_idx = st.selectbox("Query Image", range(len(pairs)), index=0, format_func=lambda i: names[i])
            with col_s:
                default_s = [i for i in range(1, min(len(pairs), nshot + 1))]
                s_indices = st.multiselect(
                    f"Support Image(s) (select {nshot})",
                    [i for i in range(len(pairs)) if i != q_idx],
                    default=default_s[:nshot],
                    format_func=lambda i: names[i],
                )

            if len(s_indices) == 0:
                st.warning("Please select at least 1 support image.")
                return

            q_pil = Image.open(pairs[q_idx][0]).convert("RGB")
            q_gt_pil = Image.open(pairs[q_idx][1]).convert("L")
            for si in s_indices:
                s_pils.append(Image.open(pairs[si][0]).convert("RGB"))
                s_mask_pils.append(Image.open(pairs[si][1]).convert("L"))
        else:
            st.markdown("Upload a **Query** image and **Support** image(s) + binary mask(s) (white=foreground, black=background).")
            u_q = st.file_uploader("Query Image", type=["png", "jpg", "jpeg"])
            u_s = st.file_uploader("Support Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            u_m = st.file_uploader("Support Mask(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if u_q and u_s and u_m and len(u_s) == len(u_m):
                q_pil = Image.open(u_q).convert("RGB")
                s_pils = [Image.open(f).convert("RGB") for f in u_s]
                s_mask_pils = [Image.open(f).convert("L") for f in u_m]
            else:
                st.info("Waiting for Query image and matching Support image(s) & mask(s)...")
                return

        # Display input task
        st.subheader("1. Input Few-Shot Task")
        in_cols = st.columns(min(len(s_pils) + 1, 5))
        with in_cols[0]:
            if q_gt_pil is not None:
                gt_np = load_mask_tensor(q_gt_pil).numpy()
                st.image(overlay_mask_np(q_pil, gt_np, color=(59, 130, 246)), caption="Query + Ground Truth (Blue)", use_container_width=True)
            else:
                st.image(q_pil.resize((400, 400)), caption="Query Image", use_container_width=True)

        for idx, (sp, sm) in enumerate(zip(s_pils[:4], s_mask_pils[:4])):
            with in_cols[idx + 1]:
                sm_np = load_mask_tensor(sm).numpy()
                st.image(overlay_mask_np(sp, sm_np, color=(16, 185, 129)), caption=f"Support #{idx+1} + Mask (Green)", use_container_width=True)

        if st.button("▶️ Run Baseline vs. Improved ABCDFSS", type="primary", use_container_width=True):
            with st.spinner("Running test-time task adaptation on Baseline & Improved models..."):
                res_base = run_single_inference(
                    backbone, device, q_pil, s_pils, s_mask_pils,
                    num_epochs=num_epochs, use_dwla=False, use_dwconv=False, use_self_rec=False, use_cos_aff=False
                )
                res_imp = run_single_inference(
                    backbone, device, q_pil, s_pils, s_mask_pils,
                    num_epochs=num_epochs, use_dwla=use_dwla, dwla_temp=dwla_temp,
                    use_dwconv=use_dwconv, use_self_rec=use_self_rec, use_cos_aff=use_cos_aff, aff_temp=aff_temp
                )

            st.subheader("2. Side-by-Side Comparison: Baseline vs. Improved")
            gt_np = load_mask_tensor(q_gt_pil).numpy() if q_gt_pil is not None else None

            col_b, col_i = st.columns(2)
            for col, title, res in [
                (col_b, "📌 Baseline (ABCDFSS - CVPR'24)", res_base),
                (col_i, "✨ Improved (ABCDFSS-Plus)", res_imp),
            ]:
                with col:
                    st.markdown(f"### {title}")
                    m1, m2, m3 = st.columns(3)
                    if gt_np is not None:
                        iou_val = compute_iou_np(res["pred"], gt_np)
                        delta_str = None
                        if title.startswith("✨"):
                            base_iou = compute_iou_np(res_base["pred"], gt_np)
                            delta_str = f"{iou_val - base_iou:+.2f}%"
                        m1.metric("Query IoU", f"{iou_val:.2f}%", delta=delta_str)
                    else:
                        m1.metric("Threshold", f"{res['thresh']:.3f}")
                    m2.metric("Pred FG Ratio", f"{res['fg_ratio']:.1f}%")
                    m3.metric("Adapt+Infer Time", f"{res['time_ms']:.0f} ms")

                    v1, v2, v3 = st.columns(3)
                    v1.image(heatmap_rgb(res["logit"]), caption="Soft Heatmap (q̂_fused)", use_container_width=True)
                    v2.image((res["pred"] * 255).astype(np.uint8), caption="Binary Mask", use_container_width=True)
                    v3.image(overlay_mask_np(q_pil, res["pred"], color=(239, 68, 68)), caption="Overlay on Query", use_container_width=True)

            # Layer weights visualization
            if res_imp["layer_weights"] is not None:
                st.subheader("3. Explainability: Layer-Wise Importance (DWLA Weights)")
                fig, ax = plt.subplots(figsize=(10, 3.2))
                layers = [f"L{i+3}" for i in range(len(res_imp["layer_weights"]))]
                x = np.arange(len(layers))
                ax.bar(x - 0.18, res_base["layer_weights"], 0.36, label="Baseline (Uniform 1/13)", color="#94a3b8")
                ax.bar(x + 0.18, res_imp["layer_weights"], 0.36, label="Improved (DWLA Dynamic)", color="#3b82f6")
                ax.set_xticks(x)
                ax.set_xticklabels(layers)
                ax.set_ylabel("Layer Weight")
                ax.set_title("ResNet50 Bottleneck Layer Weights (Low: L3-L6 | Mid: L7-L12 | High: L13-L15)")
                ax.legend()
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                st.pyplot(fig)

    with tab_bench:
        metrics_path = os.path.join(ROOT_DIR, "results", "metrics.json")
        chart_path = os.path.join(ROOT_DIR, "results", "comparison_charts.png")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if os.path.exists(chart_path):
                st.image(chart_path, caption="Benchmark Comparison: Reproduction & Improvements", use_container_width=True)
            for bname, binfo in data.get("benchmarks", {}).items():
                st.markdown(f"#### {bname} ({binfo['n_episodes']} episodes)")
                st.dataframe(binfo["variants"], use_container_width=True)
        else:
            st.info("Run `run_experiments.py` to populate benchmark results here.")


if __name__ == "__main__":
    main()
