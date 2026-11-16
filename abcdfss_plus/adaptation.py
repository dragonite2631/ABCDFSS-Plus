import os
import sys
import torch
import torch.nn.functional as F

ABCDFSS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

import core.contrastivehead as ctrutils


def compute_support_self_loss(original_features, transformed_features, augmented_smasks, k, aug):
    """
    Support Self-Reconstruction Auxiliary Loss (Improvement 3a):
    Uses the same dense affinity comparison mechanism between original and augmented
    support features to supervise pixel-level foreground/background separation.
    Memory-efficient: operates per-shot and pools spatial dims > 25x25 to avoid OOM on 4GB GPUs.
    """
    b, c, h, w = original_features.shape  # b = k * aug
    if h > 25 or w > 25:
        th, tw = min(h, 25), min(w, 25)
        q_feat = F.adaptive_avg_pool2d(original_features, (th, tw))
        s_feat = F.adaptive_avg_pool2d(transformed_features, (th, tw))
        masks = F.interpolate(augmented_smasks.unsqueeze(1).float(), size=(th, tw), mode="nearest").squeeze(1)
    else:
        th, tw = h, w
        q_feat = original_features
        s_feat = transformed_features
        masks = augmented_smasks.float()

    q_norm = F.normalize(q_feat, p=2, dim=1).permute(0, 2, 3, 1).reshape(b, th * tw, c)
    s_norm = F.normalize(s_feat, p=2, dim=1).permute(0, 2, 3, 1).reshape(b, th * tw, c)

    sim = torch.matmul(q_norm, s_norm.transpose(1, 2)) / 0.2
    att = sim.softmax(dim=-1)

    pred_map = torch.matmul(att, masks.reshape(b, th * tw, 1)).view(b, th, tw)
    pred_map = pred_map.clamp(1e-6, 1.0 - 1e-6)

    # Dice + BCE loss
    bce = F.binary_cross_entropy(pred_map, masks)
    inter = (pred_map * masks).sum(dim=(-2, -1))
    union = pred_map.sum(dim=(-2, -1)) + masks.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
    return 0.5 * bce + 0.5 * dice.mean()


def chunked_dense_info_nce_loss(original_features, transformed_features, config_nce, chunk_size=1):
    """Mathematically identical to ctrutils.dense_info_nce_loss, using gradient checkpointing per chunk only on <12GB GPUs."""
    B = transformed_features.shape[0]
    has_large_vram = (
        original_features.is_cuda
        and torch.cuda.get_device_properties(original_features.device).total_memory >= 12 * (1024 ** 3)
    )
    if B <= 2 or has_large_vram:
        return ctrutils.dense_info_nce_loss(original_features, transformed_features, config_nce)
    from torch.utils.checkpoint import checkpoint

    losses = []
    for i in range(0, B, chunk_size):
        o_c = original_features[i : i + chunk_size]
        t_c = transformed_features[i : i + chunk_size]
        l = checkpoint(lambda o, t: ctrutils.dense_info_nce_loss(o, t, config_nce), o_c, t_c, use_reentrant=False)
        losses.append(l * o_c.shape[0])
    return sum(losses) / B


def fit_adapter(
    adapter,
    mapped_qfeat_vol,
    aug_qfeat_vols,
    mapped_sfeat_vol,
    aug_sfeat_vols,
    augmented_smasks,
    config_fit,
    use_self_rec=False,
    self_rec_weight=0.5,
):
    """
    Fits an adapter (either baseline ContrastiveFeatureTransformer or SpatialContextAdapter).
    """
    f_norm = F.normalize if config_fit.normalize_after_fwd_pass else ctrutils.identity_mapping
    optimizer = config_fit.optimizer(adapter.parameters(), lr=config_fit.lr)

    for epoch in range(config_fit.num_epochs):
        # Q
        q_orig = f_norm(adapter(mapped_qfeat_vol), p=2, dim=1)
        q_aug = f_norm(adapter(aug_qfeat_vols), p=2, dim=1)
        qloss = chunked_dense_info_nce_loss(q_orig, q_aug, config_fit.nce) if config_fit.q_nceloss else 0
        if config_fit.keepvarloss:
            qloss = qloss + ctrutils.keep_var_loss(q_orig, q_aug)

        # S
        s_orig = f_norm(adapter(mapped_sfeat_vol), p=2, dim=1)
        s_aug = f_norm(adapter(aug_sfeat_vols), p=2, dim=1)
        sloss = chunked_dense_info_nce_loss(s_orig, s_aug, config_fit.nce) if config_fit.s_nceloss else 0
        if config_fit.keepvarloss:
            sloss = sloss + ctrutils.keep_var_loss(s_orig, s_aug)

        kaug = s_aug.shape[0]
        aug = aug_qfeat_vols.shape[0]
        k = kaug // aug

        if config_fit.o_t_contr_proto_loss:
            proto_loss = ctrutils.ctrstive_prototype_loss(
                s_orig, s_aug, augmented_smasks, augmented_smasks, k, aug
            )
        else:
            proto_loss = 0

        self_loss = 0
        if use_self_rec:
            self_loss = self_rec_weight * compute_support_self_loss(
                s_orig, s_aug, augmented_smasks, k, aug
            )

        loss = qloss + sloss + proto_loss + self_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
