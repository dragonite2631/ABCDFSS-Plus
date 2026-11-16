import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


class PreflightError(RuntimeError):
    """Raised when a benchmark dataset path or layout is invalid."""


@dataclass
class DatasetPreflightResult:
    benchmark: str
    configured_root: str
    resolved_root: str
    num_classes: int
    available_episodes: int
    marker_summary: str


def _resolve_candidate_root(root: str, marker_fn, benchmark: str) -> str:
    if not root or not os.path.isdir(root):
        raise PreflightError(
            f"[{benchmark}] Dataset root does not exist or is not a directory: {root!r}"
        )

    if marker_fn(root):
        return os.path.abspath(root)

    children = [
        os.path.join(root, entry)
        for entry in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, entry)) and not entry.startswith(".")
    ]
    matching = [child for child in children if marker_fn(child)]
    if len(matching) == 1:
        return os.path.abspath(matching[0])
    if len(matching) > 1:
        raise PreflightError(
            f"[{benchmark}] Ambiguous nested directories under {root!r}: {matching}. "
            "Set the exact dataset root explicitly."
        )

    raise PreflightError(
        f"[{benchmark}] Expected dataset layout not found under {root!r}. "
        f"Top-level directories found: {[os.path.basename(c) for c in children]}"
    )


def _validate_deepglobe(root: str) -> DatasetPreflightResult:
    categories = ["1", "2", "3", "4", "5", "6"]

    def has_markers(path: str) -> bool:
        return all(
            os.path.isdir(os.path.join(path, c, "test", "origin"))
            and os.path.isdir(os.path.join(path, c, "test", "groundtruth"))
            for c in categories
        )

    resolved = _resolve_candidate_root(root, has_markers, "deepglobe")
    total_pairs = 0
    for c in categories:
        origin_dir = os.path.join(resolved, c, "test", "origin")
        gt_dir = os.path.join(resolved, c, "test", "groundtruth")
        imgs = sorted(glob.glob(os.path.join(origin_dir, "*.jpg")))
        valid_c = sum(
            1
            for img_path in imgs
            if os.path.isfile(os.path.join(gt_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.png"))
        )
        if valid_c == 0:
            raise PreflightError(
                f"[deepglobe] Class {c!r} has 0 valid image/mask pairs in {origin_dir!r}"
            )
        total_pairs += valid_c

    return DatasetPreflightResult(
        benchmark="deepglobe",
        configured_root=root,
        resolved_root=resolved,
        num_classes=len(categories),
        available_episodes=total_pairs,
        marker_summary=f"6 classes, {total_pairs} image/mask pairs",
    )


def _validate_isic(root: str) -> DatasetPreflightResult:
    categories = ["1", "2", "3"]

    def has_markers(path: str) -> bool:
        return os.path.isdir(
            os.path.join(path, "ISIC2018_Task1_Training_GroundTruth")
        ) and all(
            os.path.isdir(os.path.join(path, "ISIC2018_Task1-2_Training_Input", c))
            for c in categories
        )

    resolved = _resolve_candidate_root(root, has_markers, "isic")
    gt_dir = os.path.join(resolved, "ISIC2018_Task1_Training_GroundTruth")
    total_pairs = 0
    for c in categories:
        img_dir = os.path.join(resolved, "ISIC2018_Task1-2_Training_Input", c)
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        valid_c = sum(
            1
            for img_path in imgs
            if os.path.isfile(os.path.join(gt_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_segmentation.png"))
        )
        if valid_c == 0:
            raise PreflightError(
                f"[isic] Class {c!r} has 0 valid image/mask pairs in {img_dir!r}"
            )
        total_pairs += valid_c

    return DatasetPreflightResult(
        benchmark="isic",
        configured_root=root,
        resolved_root=resolved,
        num_classes=len(categories),
        available_episodes=total_pairs,
        marker_summary=f"3 classes, {total_pairs} image/mask pairs",
    )


def _validate_lung(root: str) -> DatasetPreflightResult:
    def has_markers(path: str) -> bool:
        return os.path.isdir(os.path.join(path, "CXR_png")) and os.path.isdir(
            os.path.join(path, "masks")
        )

    resolved = _resolve_candidate_root(root, has_markers, "lung")
    cxr_dir = os.path.join(resolved, "CXR_png")
    masks_dir = os.path.join(resolved, "masks")
    masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    valid_pairs = 0
    for mask_path in masks:
        base = os.path.basename(mask_path)
        img_name = base[:-9] + ".png" if base.endswith("_mask.png") else base
        if os.path.isfile(os.path.join(cxr_dir, img_name)):
            valid_pairs += 1
    if valid_pairs == 0:
        raise PreflightError(
            f"[lung] Found 0 valid CXR/mask pairs under {resolved!r}"
        )

    return DatasetPreflightResult(
        benchmark="lung",
        configured_root=root,
        resolved_root=resolved,
        num_classes=1,
        available_episodes=valid_pairs,
        marker_summary=f"1 class, {valid_pairs} CXR/mask pairs",
    )


def _validate_fss(root: str, abcdfss_dir: str) -> DatasetPreflightResult:
    split_file = os.path.join(abcdfss_dir, "data", "splits", "fss", "test.txt")
    if not os.path.isfile(split_file):
        raise PreflightError(f"[fss] Missing predefined split file: {split_file!r}")

    with open(split_file, "r", encoding="utf-8") as f:
        categories = [line.strip() for line in f if line.strip()]

    def has_markers(path: str) -> bool:
        return os.path.isdir(os.path.join(path, "FSS-1000"))

    resolved = _resolve_candidate_root(root, has_markers, "fss")
    fss_dir = os.path.join(resolved, "FSS-1000")

    valid_classes = 0
    total_pairs = 0
    for cat in categories:
        cat_dir = os.path.join(fss_dir, cat)
        if not os.path.isdir(cat_dir):
            raise PreflightError(
                f"[fss] Missing test split category folder {cat!r} in {fss_dir!r}"
            )
        imgs = sorted(glob.glob(os.path.join(cat_dir, "*.jpg")))
        cat_pairs = sum(
            1
            for img_path in imgs
            if os.path.isfile(os.path.splitext(img_path)[0] + ".png")
        )
        if cat_pairs == 0:
            raise PreflightError(
                f"[fss] Category {cat!r} has 0 valid jpg/png pairs in {cat_dir!r}"
            )
        valid_classes += 1
        total_pairs += cat_pairs

    return DatasetPreflightResult(
        benchmark="fss",
        configured_root=root,
        resolved_root=resolved,
        num_classes=valid_classes,
        available_episodes=total_pairs,
        marker_summary=f"{valid_classes} test classes, {total_pairs} image/mask pairs",
    )


def _validate_suim(root: str) -> DatasetPreflightResult:
    categories = ["FV", "HD", "PF", "RI", "RO", "SR", "WR"]

    def has_markers(path: str) -> bool:
        return os.path.isdir(os.path.join(path, "images")) and os.path.isdir(
            os.path.join(path, "masks")
        )

    resolved = _resolve_candidate_root(root, has_markers, "suim")
    img_dir = os.path.join(resolved, "images")
    masks_root = os.path.join(resolved, "masks")

    total_pairs = 0
    for c in categories:
        c_mask_dir = os.path.join(masks_root, c)
        if not os.path.isdir(c_mask_dir):
            raise PreflightError(
                f"[suim] Missing class mask directory {c_mask_dir!r}"
            )
        masks = sorted(glob.glob(os.path.join(c_mask_dir, "*.bmp"))) + sorted(
            glob.glob(os.path.join(c_mask_dir, "*.png"))
        )
        valid_c = sum(
            1
            for mask_path in masks
            if os.path.isfile(os.path.join(img_dir, f"{os.path.splitext(os.path.basename(mask_path))[0]}.jpg"))
        )
        if valid_c == 0:
            raise PreflightError(
                f"[suim] Class {c!r} has 0 valid image/mask pairs in {c_mask_dir!r}"
            )
        total_pairs += valid_c

    return DatasetPreflightResult(
        benchmark="suim",
        configured_root=root,
        resolved_root=resolved,
        num_classes=len(categories),
        available_episodes=total_pairs,
        marker_summary=f"7 classes, {total_pairs} image/mask pairs",
    )


def validate_all_datasets(
    dataset_roots: Dict[str, str],
    abcdfss_dir: str,
    benchmarks: Optional[List[str]] = None,
) -> Dict[str, DatasetPreflightResult]:
    validators = {
        "deepglobe": lambda r: _validate_deepglobe(r),
        "isic": lambda r: _validate_isic(r),
        "lung": lambda r: _validate_lung(r),
        "fss": lambda r: _validate_fss(r, abcdfss_dir),
        "suim": lambda r: _validate_suim(r),
    }
    target_benchmarks = benchmarks or list(validators.keys())
    results: Dict[str, DatasetPreflightResult] = {}
    for bname in target_benchmarks:
        if bname not in validators:
            raise PreflightError(f"Unknown benchmark key: {bname!r}")
        if bname not in dataset_roots:
            raise PreflightError(f"Missing dataset root configuration for {bname!r}")
        results[bname] = validators[bname](dataset_roots[bname])
    return results
