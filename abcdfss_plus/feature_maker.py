import os
import sys
import torch

ABCDFSS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ABCDFSS")
if ABCDFSS_DIR not in sys.path:
    sys.path.insert(0, ABCDFSS_DIR)

from utils import segutils
import core.contrastivehead as ctrutils
from abcdfss_plus.adapters import SpatialContextAdapter
from abcdfss_plus.adaptation import fit_adapter


def make_plus_config(
    fit_every_episode=False,
    q_nceloss=True,
    s_nceloss=True,
    keepvarloss=True,
    o_t_contr_proto_loss=True,
    num_epochs=25,
    lr=1e-2,
):
    """Creates the standard ContrastiveConfig used by ABCDFSS-Plus."""
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


class ImprovedCTrBuilder(ctrutils.CTrBuilder):
    """
    Extension of ABCDFSS CTrBuilder supporting SpatialContextAdapter and Support Self-Reconstruction Loss.
    """

    def __init__(self, config, use_dwconv_adapter=False, use_self_rec=False, self_rec_weight=0.5):
        super().__init__(config)
        self.use_dwconv_adapter = use_dwconv_adapter
        self.use_self_rec = use_self_rec
        self.self_rec_weight = self_rec_weight

    def buildContrastiveTransformers(
        self,
        qfeat_alllayers,
        sfeat_alllayers,
        query_feats_aug,
        support_feats_aug,
        supp_aug_mask,
        s_mask=None,
    ):
        contrastive_transformers = []
        l0 = self.config.featext.l0
        s_aug_mask = supp_aug_mask.view(-1, *supp_aug_mask.shape[-2:])
        device = qfeat_alllayers[l0].device

        for qfeat, sfeat, qfeataug, sfeataug in zip(
            qfeat_alllayers[l0:], sfeat_alllayers[l0:], query_feats_aug[l0:], support_feats_aug[l0:]
        ):
            bsz, k, aug, ch, h, w = sfeataug.shape
            qfeat_v = qfeat.view(-1, *qfeat.shape[-3:])
            sfeat_v = sfeat.view(-1, *sfeat.shape[-3:])
            qfeataug_v = qfeataug.view(-1, *qfeataug.shape[-3:])
            sfeataug_v = sfeataug.view(-1, *sfeataug.shape[-3:])

            if self.use_dwconv_adapter:
                adapter = SpatialContextAdapter(in_channels=ch, config_model=self.config.model).to(device)
            else:
                adapter = ctrutils.ContrastiveFeatureTransformer(
                    in_channels=ch, config_model=self.config.model
                ).to(device)

            mapped_qfeat = self.augmentator.applyAffines(qfeat_v).view(-1, *qfeat_v.shape[-3:])
            mapped_sfeat = self.augmentator.applyAffines(sfeat_v).view(-1, *sfeat_v.shape[-3:])

            fit_adapter(
                adapter,
                mapped_qfeat,
                qfeataug_v,
                mapped_sfeat,
                sfeataug_v,
                segutils.downsample_mask(s_aug_mask, h, w),
                self.config.fitting,
                use_self_rec=self.use_self_rec,
                self_rec_weight=self.self_rec_weight,
            )
            contrastive_transformers.append(adapter)
            if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory < 8 * (1024 ** 3):
                torch.cuda.empty_cache()

        return contrastive_transformers


class ImprovedFeatureMaker(ctrutils.FeatureMaker):
    def __init__(
        self,
        feat_extraction_method,
        class_ids,
        config,
        use_dwconv_adapter=False,
        use_self_rec=False,
        self_rec_weight=0.5,
    ):
        super().__init__(feat_extraction_method, class_ids, config)
        self.c_trs = {
            ctr: ImprovedCTrBuilder(
                config,
                use_dwconv_adapter=use_dwconv_adapter,
                use_self_rec=use_self_rec,
                self_rec_weight=self_rec_weight,
            )
            for ctr in class_ids
        }
