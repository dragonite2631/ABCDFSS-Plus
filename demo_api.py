import base64
import glob
import io
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ABCDFSS_DIR = os.path.join(ROOT_DIR, "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

from core.backbone import Backbone
from utils import commonutils as utils
from utils import segutils
from abcdfss_plus import (
    ImprovedDAMatComparison,
    ImprovedFeatureMaker,
    compute_layer_discriminability,
    make_plus_config,
)

app = FastAPI(title="ABCDFSS-Plus Interactive Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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

LAYER_META = [
    (3, "L3 (layer2.0)", "Low"),
    (4, "L4 (layer2.1)", "Low"),
    (5, "L5 (layer2.2)", "Low"),
    (6, "L6 (layer2.3)", "Low"),
    (7, "L7 (layer3.0)", "Mid"),
    (8, "L8 (layer3.1)", "Mid"),
    (9, "L9 (layer3.2)", "Mid"),
    (10, "L10 (layer3.3)", "Mid"),
    (11, "L11 (layer3.4)", "Mid"),
    (12, "L12 (layer3.5)", "Mid"),
    (13, "L13 (layer4.0)", "High"),
    (14, "L14 (layer4.1)", "High"),
    (15, "L15 (layer4.2)", "High"),
]

_BACKBONE_CACHE = {"model": None, "device": None}


def get_backbone():
    if _BACKBONE_CACHE["model"] is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        bb = Backbone("resnet50").to(device)
        bb.eval()
        _BACKBONE_CACHE["model"] = bb
        _BACKBONE_CACHE["device"] = device
    return _BACKBONE_CACHE["model"], _BACKBONE_CACHE["device"]


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _arr_to_heatmap_b64(arr_2d: np.ndarray, cmap_name: str = "inferno", size=(200, 200)) -> str:
    a = arr_2d.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn > 1e-6:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    rgba = (matplotlib.colormaps[cmap_name](a) * 255).astype(np.uint8)
    img = Image.fromarray(rgba[:, :, :3]).resize(size, Image.Resampling.BILINEAR)
    return _pil_to_b64(img)


def _overlay_b64(base_rgb: Image.Image, mask_2d: np.ndarray, color=(245, 158, 11), alpha=0.55) -> str:
    import cv2

    base = np.array(base_rgb.resize((400, 400))).astype(np.float32)
    m_bin = (mask_2d > 0.5).astype(np.uint8)
    m = m_bin.astype(np.float32)[:, :, None]
    c = np.array(color, dtype=np.float32)[None, None, :]
    blended = (base * (1.0 - alpha * m) + c * (alpha * m)).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, 2)
    return _pil_to_b64(Image.fromarray(blended))


def _mask_b64(mask_2d: np.ndarray) -> str:
    arr = ((mask_2d > 0.5).astype(np.uint8) * 255)
    return _pil_to_b64(Image.fromarray(arr).convert("RGB"))


def _load_mask_tensor(mask_pil: Image.Image, size=(400, 400)) -> torch.Tensor:
    arr = np.array(mask_pil.convert("L"))
    t = torch.tensor(arr, dtype=torch.float32)
    t = (t >= 128).float()
    return F.interpolate(t.unsqueeze(0).unsqueeze(0), size=size, mode="nearest").squeeze(0).squeeze(0)


def _compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0.5
    g = gt > 0.5
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return round(float(inter / union * 100.0), 2) if union > 0 else 0.0


def _scan_lung_items() -> List[dict]:
    cxr_dir = os.path.join(ROOT_DIR, "datasets", "lung", "CXR_png")
    mask_dir = os.path.join(ROOT_DIR, "datasets", "lung", "masks")
    if not (os.path.isdir(cxr_dir) and os.path.isdir(mask_dir)):
        return []
    items = []
    for mp in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        base = os.path.basename(mp)
        img_name = base[:-9] + ".png" if base.endswith("_mask.png") else base
        ip = os.path.join(cxr_dir, img_name)
        if os.path.isfile(ip):
            items.append({
                "id": os.path.splitext(img_name)[0],
                "img_rel": os.path.relpath(ip, ROOT_DIR).replace("\\", "/"),
                "mask_rel": os.path.relpath(mp, ROOT_DIR).replace("\\", "/"),
            })
    return items


def _scan_suim_items(cls_code: str) -> List[dict]:
    img_dir = os.path.join(ROOT_DIR, "datasets", "suim", "images")
    mask_dir = os.path.join(ROOT_DIR, "datasets", "suim", "masks", cls_code)
    if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
        return []
    items = []
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.bmp"))) + sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    for mp in masks:
        stem = os.path.splitext(os.path.basename(mp))[0]
        ip = os.path.join(img_dir, f"{stem}.jpg")
        if os.path.isfile(ip):
            items.append({
                "id": stem,
                "img_rel": os.path.relpath(ip, ROOT_DIR).replace("\\", "/"),
                "mask_rel": os.path.relpath(mp, ROOT_DIR).replace("\\", "/"),
            })
    return items


def _run_single_variant(
    backbone,
    device: torch.device,
    q_tensor: torch.Tensor,
    s_tensors: torch.Tensor,
    s_masks: torch.Tensor,
    q_pil: Image.Image,
    q_gt_np: Optional[np.ndarray],
    is_plus: bool,
    num_epochs: int,
    seed: int = 42,
) -> dict:
    utils.fix_randseed(seed)
    config = make_plus_config(fit_every_episode=True, num_epochs=num_epochs)
    feat_maker = ImprovedFeatureMaker(
        backbone.extract_feats,
        [0],
        config,
        use_dwconv_adapter=is_plus,
        use_self_rec=is_plus,
    )
    damat_comp = ImprovedDAMatComparison(
        use_dwla=is_plus,
        use_cosine_affinity=is_plus,
    )

    t0 = time.time()
    q_feat_t, s_feat_t = feat_maker.taskAdapt(q_tensor, s_tensors, s_masks, 0)
    q_pred_coarses_t, deltas = damat_comp.calc_q_pred_coarses_and_deltas(q_feat_t, s_feat_t, s_masks, l0=3)

    if is_plus and len(deltas) > 0:
        delta_t = torch.tensor(deltas, device=q_pred_coarses_t.device, dtype=torch.float32)
        weights = F.softmax(delta_t / damat_comp.dwla_temp, dim=0)
        layer_weights = weights.detach().cpu().tolist()
        logit_mask = (q_pred_coarses_t * weights.view(1, -1, 1, 1)).sum(dim=1)
    else:
        num_layers = q_pred_coarses_t.shape[1]
        layer_weights = [1.0 / num_layers] * num_layers
        # Compute deltas for display even in baseline
        deltas = []
        for sft in s_feat_t[3:]:
            hs, ws = sft.shape[-2:]
            sm_row = torch.cat([segutils.downsample_mask(m, hs, ws) for m in s_masks.unbind(1)], -1)
            deltas.append(compute_layer_discriminability(sft.detach(), sm_row))
        logit_mask = q_pred_coarses_t.mean(1)

    logit_mask = segutils.downsample_mask(logit_mask, *s_masks.shape[-2:])
    thresh = segutils.calcthresh(logit_mask, s_masks, "pred_mean")
    pred_mask = (logit_mask > thresh).float()
    elapsed = round(time.time() - t0, 2)

    pred_np = pred_mask[0].detach().cpu().numpy()
    logit_np = logit_mask[0].detach().cpu().numpy()

    layers_info = []
    for idx, (layer_id, label, stage) in enumerate(LAYER_META):
        qf = q_feat_t[layer_id].detach()[0]  # [c, h, w]
        feat_energy = torch.norm(qf, p=2, dim=0).cpu().numpy()
        coarse_map = q_pred_coarses_t[0, idx].detach().cpu().numpy()
        layers_info.append({
            "layer_id": layer_id,
            "label": label,
            "stage": stage,
            "channels": int(qf.shape[0]),
            "spatial": f"{qf.shape[1]}x{qf.shape[2]}",
            "delta": round(float(deltas[idx]) * 100.0, 2),
            "weight": round(float(layer_weights[idx]) * 100.0, 2),
            "coarse_heatmap_b64": _arr_to_heatmap_b64(coarse_map, "viridis", size=(160, 160)),
            "feature_norm_b64": _arr_to_heatmap_b64(feat_energy, "magma", size=(160, 160)),
        })

    color = (239, 68, 68) if is_plus else (217, 70, 239)
    return {
        "name": "ABCDFSS-Plus" if is_plus else "Baseline ABCDFSS",
        "elapsed_sec": elapsed,
        "threshold": round(float(thresh), 4),
        "iou": _compute_iou(pred_np, q_gt_np) if q_gt_np is not None else None,
        "pred_mask_b64": _mask_b64(pred_np),
        "overlay_b64": _overlay_b64(q_pil, pred_np, color=color),
        "heatmap_b64": _arr_to_heatmap_b64(logit_np, "inferno", size=(400, 400)),
        "layers": layers_info,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.post("/api/pick-file")
def pick_file(title: str = "ChÃ¡Â»Ân Ã¡ÂºÂ£nh tÃ¡Â»Â« demo_samples"):
    initial_dir = os.path.join(ROOT_DIR, "demo_samples")
    if not os.path.isdir(initial_dir):
        initial_dir = ROOT_DIR
    code = (
        "import tkinter as tk; from tkinter import filedialog; "
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); "
        f"p = filedialog.askopenfilename(initialdir={initial_dir!r}, title={title!r}, "
        "filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp'), ('All files', '*.*')]); "
        "r.destroy(); print(p)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    selected = proc.stdout.strip()
    if not selected or not os.path.isfile(selected):
        return {"path": None, "name": None, "preview_b64": None, "mask_path": None}

    # Auto-detect the paired mask using the demo_samples naming rule: <stem>_img.ext -> <stem>_mask.png
    mask_path = None
    base = os.path.basename(selected)
    stem, _ext = os.path.splitext(base)
    if stem.endswith("_img"):
        cand = os.path.join(os.path.dirname(selected), stem[:-4] + "_mask.png")
        if os.path.isfile(cand):
            mask_path = cand

    img_pil = Image.open(selected).convert("RGB")
    if mask_path:
        mask_np = _load_mask_tensor(Image.open(mask_path), size=(400, 400)).numpy()
        preview = _overlay_b64(img_pil, mask_np, color=(245, 158, 11))
    else:
        preview = _pil_to_b64(img_pil.resize((220, 220)))
    return {
        "path": selected,
        "name": base,
        "preview_b64": preview,
        "mask_path": mask_path,
    }


@app.post("/api/open-demo-folder")
def open_demo_folder(subfolder: Optional[str] = None):
    target = os.path.join(ROOT_DIR, "demo_samples")
    if subfolder:
        cand = os.path.abspath(os.path.join(target, subfolder))
        if cand.startswith(os.path.abspath(target)) and os.path.isdir(cand):
            target = cand
    if os.path.isdir(target) and hasattr(os, "startfile"):
        os.startfile(target)
    return {"opened": target}


@app.get("/api/demo-samples")
def list_demo_samples():
    demo_root = os.path.join(ROOT_DIR, "demo_samples")
    if not os.path.isdir(demo_root):
        return {"groups": {}}
    groups = {}
    for folder in sorted(os.listdir(demo_root)):
        fpath = os.path.join(demo_root, folder)
        if not os.path.isdir(fpath):
            continue
        items = []
        for mp in sorted(glob.glob(os.path.join(fpath, "*_mask.png"))):
            stem = os.path.basename(mp)[:-9]  # strip _mask.png
            ip = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = os.path.join(fpath, f"{stem}_img{ext}")
                if os.path.isfile(cand):
                    ip = cand
                    break
            if ip is None:
                continue
            img_pil = Image.open(ip).convert("RGB")
            mask_np = _load_mask_tensor(Image.open(mp), size=(400, 400)).numpy()
            items.append({
                "id": stem,
                "img_rel": os.path.relpath(ip, ROOT_DIR).replace("\\", "/"),
                "mask_rel": os.path.relpath(mp, ROOT_DIR).replace("\\", "/"),
                "img_abs": ip,
                "mask_abs": mp,
                "thumb_b64": _pil_to_b64(img_pil.resize((220, 220))),
                "overlay_b64": _overlay_b64(img_pil, mask_np, color=(245, 158, 11)),
            })
        if items:
            groups[folder] = items
    return {"groups": groups}


@app.get("/api/datasets")
def list_datasets():
    lung_items = _scan_lung_items()
    suim_classes = {}
    for code, name in SUIM_CLASSES.items():
        items = _scan_suim_items(code)
        if items:
            suim_classes[code] = {
                "name": name,
                "count": len(items),
                "samples": items[:24],
            }
    return {
        "datasets": {
            "lung": {
                "name": "Chest X-ray Lung (Montgomery)",
                "count": len(lung_items),
                "classes": {"1": {"name": "Lung (1)", "count": len(lung_items), "samples": lung_items[:24]}},
            },
            "suim": {
                "name": "SUIM Underwater Imagery",
                "count": sum(v["count"] for v in suim_classes.values()),
                "classes": suim_classes,
            },
        }
    }


@app.get("/api/image")
def get_image(rel_path: str):
    abs_path = os.path.abspath(os.path.join(ROOT_DIR, rel_path))
    datasets_root = os.path.abspath(os.path.join(ROOT_DIR, "datasets"))
    if not abs_path.startswith(datasets_root) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image not found")
    # Convert .bmp or mask to PNG in-memory for universal browser display
    img = Image.open(abs_path).convert("RGB").resize((400, 400))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return JSONResponse({"data_url": _pil_to_b64(img)})


@app.get("/api/benchmark-results")
def get_benchmark_results():
    metrics_path = os.path.join(ROOT_DIR, "results", "metrics.json")
    if not os.path.isfile(metrics_path):
        return {"benchmarks": {}}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/compare")
async def compare_episode(
    mode: str = Form("dataset"),
    dataset: str = Form("lung"),
    class_code: str = Form("1"),
    support_indices: str = Form("0"),
    query_index: int = Form(1),
    num_epochs: int = Form(25),
    support_img_rel: Optional[str] = Form(None),
    support_mask_rel: Optional[str] = Form(None),
    query_img_rel: Optional[str] = Form(None),
    query_mask_rel: Optional[str] = Form(None),
    support_img_path: Optional[str] = Form(None),
    support_mask_path: Optional[str] = Form(None),
    query_img_path: Optional[str] = Form(None),
    query_mask_path: Optional[str] = Form(None),
    support_img_file: Optional[UploadFile] = File(None),
    support_mask_file: Optional[UploadFile] = File(None),
    query_img_file: Optional[UploadFile] = File(None),
    query_mask_file: Optional[UploadFile] = File(None),
):
    backbone, device = get_backbone()
    s_pils: List[Image.Image] = []
    s_mask_tensors: List[torch.Tensor] = []
    q_gt_np: Optional[np.ndarray] = None

    if mode == "upload":
        if support_img_path and support_mask_path and query_img_path:
            s_pils.append(Image.open(support_img_path).convert("RGB"))
            s_mask_tensors.append(_load_mask_tensor(Image.open(support_mask_path)))
            q_pil = Image.open(query_img_path).convert("RGB")
            if query_mask_path:
                q_gt_np = _load_mask_tensor(Image.open(query_mask_path)).numpy()
        elif support_img_file and support_mask_file and query_img_file:
            s_pils.append(Image.open(io.BytesIO(await support_img_file.read())).convert("RGB"))
            s_m_pil = Image.open(io.BytesIO(await support_mask_file.read()))
            s_mask_tensors.append(_load_mask_tensor(s_m_pil))
            q_pil = Image.open(io.BytesIO(await query_img_file.read())).convert("RGB")
            if query_mask_file is not None:
                q_gt_np = _load_mask_tensor(Image.open(io.BytesIO(await query_mask_file.read()))).numpy()
        else:
            raise HTTPException(status_code=400, detail="Missing uploaded support/query files")
    elif mode == "demo":
        if not (support_img_rel and support_mask_rel and query_img_rel):
            raise HTTPException(status_code=400, detail="Missing demo sample paths")
        s_pils.append(Image.open(os.path.join(ROOT_DIR, support_img_rel)).convert("RGB"))
        s_mask_tensors.append(_load_mask_tensor(Image.open(os.path.join(ROOT_DIR, support_mask_rel))))
        q_pil = Image.open(os.path.join(ROOT_DIR, query_img_rel)).convert("RGB")
        if query_mask_rel:
            q_gt_np = _load_mask_tensor(Image.open(os.path.join(ROOT_DIR, query_mask_rel))).numpy()
    else:
        items = _scan_lung_items() if dataset == "lung" else _scan_suim_items(class_code)
        if len(items) < 2:
            raise HTTPException(status_code=404, detail="Not enough samples in dataset")
        s_idxs = [int(x.strip()) for x in support_indices.split(",") if x.strip()]
        if not s_idxs:
            s_idxs = [0]
        for idx in s_idxs:
            it = items[idx % len(items)]
            s_pils.append(Image.open(os.path.join(ROOT_DIR, it["img_rel"])).convert("RGB"))
            s_mask_tensors.append(_load_mask_tensor(Image.open(os.path.join(ROOT_DIR, it["mask_rel"]))))
        q_it = items[query_index % len(items)]
        q_pil = Image.open(os.path.join(ROOT_DIR, q_it["img_rel"])).convert("RGB")
        q_gt_np = _load_mask_tensor(Image.open(os.path.join(ROOT_DIR, q_it["mask_rel"]))).numpy()

    q_tensor = IMG_TRANSFORM(q_pil).unsqueeze(0).to(device)
    s_tensors = torch.stack([IMG_TRANSFORM(p) for p in s_pils]).unsqueeze(0).to(device)
    s_masks = torch.stack(s_mask_tensors).unsqueeze(0).to(device)

    baseline_res = _run_single_variant(
        backbone, device, q_tensor, s_tensors, s_masks, q_pil, q_gt_np, is_plus=False, num_epochs=num_epochs
    )
    plus_res = _run_single_variant(
        backbone, device, q_tensor, s_tensors, s_masks, q_pil, q_gt_np, is_plus=True, num_epochs=num_epochs
    )

    support_previews = [
        {
            "image_b64": _pil_to_b64(sp.resize((400, 400))),
            "overlay_b64": _overlay_b64(sp, sm.numpy(), color=(245, 158, 11)),
        }
        for sp, sm in zip(s_pils, s_mask_tensors)
    ]

    return {
        "support": support_previews,
        "query": {
            "image_b64": _pil_to_b64(q_pil.resize((400, 400))),
            "gt_overlay_b64": _overlay_b64(q_pil, q_gt_np, color=(245, 158, 11)) if q_gt_np is not None else None,
            "gt_mask_b64": _mask_b64(q_gt_np) if q_gt_np is not None else None,
        },
        "baseline": baseline_res,
        "plus": plus_res,
    }


DIST_DIR = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.isdir(DIST_DIR):
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
