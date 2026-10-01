"""特征计算模块

提供相似度聚合特征和手工特征的计算功能。
"""

from nallm.feature.compute import (
    SIM_METRIC_CONFIG,
    CORE_FEATURE_INDICES,
    SIM_FEATURE_DIM,
    HAND_FEATURE_DIM,
    aggregate_sim_features,
    compute_hand_features,
)

__all__ = [
    "SIM_METRIC_CONFIG",
    "CORE_FEATURE_INDICES",
    "SIM_FEATURE_DIM",
    "HAND_FEATURE_DIM",
    "aggregate_sim_features",
    "compute_hand_features",
]
