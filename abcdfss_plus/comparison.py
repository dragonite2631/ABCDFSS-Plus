import os
import sys
import torch
import torch.nn.functional as F

ABCDFSS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

from utils import segutils
import core.denseaffinity as dautils


def compute_layer_discriminability(s_feat_layer, s_mask_downsampled):
    """
    Computes intra-support discriminability delta_l = sim(FG<->FG) - sim(FG<->BG)
    using cosine similarity (Eq. 20, 21, 26 in Appendix C).
    s_feat_layer: [bsz, k, c, h, w] -> flattened across (k, h, w)
    s_mask_downsampled: [bsz, k, h, w] or horizontally concatenated [bsz, h, k*w]
    """
    bsz, k, c, h, w = s_feat_layer.shape
    feat_flat = F.normalize(s_feat_layer.permute(0, 1, 3, 4, 2).reshape(bsz, k * h * w, c), p=2, dim=-1)
    if k > 1 and s_mask_downsampled.dim() == 3 and s_mask_downsampled.shape == (bsz, h, k * w):
        mask_flat = s_mask_downsampled.view(bsz, h, k, w).permute(0, 2, 1, 3).reshape(bsz, k * h * w)
    else:
        mask_flat = s_mask_downsampled.reshape(bsz, k * h * w)

    fg_idx = mask_flat[0] > 0.5
    bg_idx = ~fg_idx
    if fg_idx.sum() == 0 or bg_idx.sum() == 0:
        return 0.0

    fg_feats = feat_flat[0, fg_idx]  # [N_fg, c]
    bg_feats = feat_flat[0, bg_idx]  # [N_bg, c]

    # Mean cosine similarity is equivalent to dot product of mean normalized vectors!
    fg_mean = fg_feats.mean(dim=0)
    bg_mean = bg_feats.mean(dim=0)
    sim_fg_fg = torch.dot(fg_mean, fg_mean).item()
    sim_fg_bg = torch.dot(fg_mean, bg_mean).item()
    return sim_fg_fg - sim_fg_bg


class ImprovedDAMatComparison:
    """
    Enhanced Dense Affinity Comparison supporting:
    - Improvement 1: Discriminability-Weighted Layer Aggregation (DWLA)
    - Improvement 3b: Cosine-normalized Dense Affinity with temperature scaling
    """

    def __init__(self, use_dwla=False, dwla_temp=0.1, use_cosine_affinity=False, aff_temp=0.15):
        self.use_dwla = use_dwla
        self.dwla_temp = dwla_temp
        self.use_cosine_affinity = use_cosine_affinity
        self.aff_temp = aff_temp
        self.last_layer_weights = None
        self.last_layer_deltas = None

    def calc_q_pred_coarses_and_deltas(self, q_feat_t, s_feat_t, s_mask, l0=3):
        q_pred_coarses = []
        deltas = []
        h0, w0 = q_feat_t[l0].shape[-2:]

        for qft, sft in zip(q_feat_t[l0:], s_feat_t[l0:]):
            qft, sft = qft.detach(), sft.detach()
            bsz, c, hq, wq = qft.shape
            hs, ws = sft.shape[-2:]

            sft_row = torch.cat(sft.unbind(1), -1)  # [bsz, c, h, w*k]
            smasks_downsampled = [segutils.downsample_mask(m, hs, ws) for m in s_mask.unbind(1)]
            smask_row = torch.cat(smasks_downsampled, -1)

            if self.use_dwla:
                smask_stacked = torch.stack(smasks_downsampled, dim=1)  # [bsz, k, h, w]
                delta = compute_layer_discriminability(sft, smask_stacked)
                deltas.append(delta)

            if self.use_cosine_affinity:
                q_norm = F.normalize(qft, p=2, dim=1)
                s_norm = F.normalize(sft_row, p=2, dim=1)
                q_v = q_norm.permute(0, 2, 3, 1).reshape(bsz, hq * wq, c)
                s_v = s_norm.permute(0, 2, 3, 1).reshape(bsz, hs * ws * sft.shape[1], c)
                damat = (torch.matmul(q_v, s_v.transpose(1, 2)) / self.aff_temp).softmax(dim=-1)
            else:
                damat = dautils.buildDenseAffinityMat(qft, sft_row)

            filtered = dautils.filterDenseAffinityMap(damat, smask_row)
            q_pred_coarse = dautils.upsample(filtered.view(bsz, 1, hq, wq), h0, w0).squeeze(1)
            q_pred_coarses.append(q_pred_coarse)

        return torch.stack(q_pred_coarses, dim=1), deltas

    def forward(self, q_feat_t, s_feat_t, s_mask, upsample=True):
        q_pred_coarses_t, deltas = self.calc_q_pred_coarses_and_deltas(q_feat_t, s_feat_t, s_mask)

        if self.use_dwla and len(deltas) > 0:
            delta_t = torch.tensor(deltas, device=q_pred_coarses_t.device, dtype=torch.float32)
            weights = F.softmax(delta_t / self.dwla_temp, dim=0)
            self.last_layer_weights = weights.detach().cpu().tolist()
            self.last_layer_deltas = deltas
            logit_mask = (q_pred_coarses_t * weights.view(1, -1, 1, 1)).sum(dim=1)
        else:
            num_layers = q_pred_coarses_t.shape[1]
            self.last_layer_weights = [1.0 / num_layers] * num_layers
            self.last_layer_deltas = deltas
            logit_mask = q_pred_coarses_t.mean(1)

        if upsample:
            logit_mask = segutils.downsample_mask(logit_mask, *s_mask.shape[-2:])
        return logit_mask
