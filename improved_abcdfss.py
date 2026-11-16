"""Compatibility re-export module for ABCDFSS-Plus components."""

from abcdfss_plus.adapters import SpatialContextAdapter
from abcdfss_plus.adaptation import (
    chunked_dense_info_nce_loss,
    compute_support_self_loss,
    fit_adapter,
)
from abcdfss_plus.comparison import (
    ImprovedDAMatComparison,
    compute_layer_discriminability,
)
from abcdfss_plus.feature_maker import (
    ImprovedCTrBuilder,
    ImprovedFeatureMaker,
    make_plus_config,
)

__all__ = [
    "SpatialContextAdapter",
    "compute_support_self_loss",
    "chunked_dense_info_nce_loss",
    "fit_adapter",
    "compute_layer_discriminability",
    "ImprovedDAMatComparison",
    "ImprovedCTrBuilder",
    "ImprovedFeatureMaker",
    "make_plus_config",
]
