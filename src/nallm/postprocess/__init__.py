"""模型辅助后处理模块

提供 LightGBM 模型辅助的消歧结果后处理功能：
- 规则 A: LLM 分配 + 模型概率低 → 纠正为 none（减少误判）
- 规则 B: LLM 返回 none + 模型概率高 → 分配候选（减少漏判）
"""

from nallm.postprocess.config import THETA_A, THETA_B, MODEL_PATH
from nallm.postprocess.model import DisambiguationModel
from nallm.postprocess.feature_compute import FEATURE_DIM, FeatureLookup
from nallm.postprocess.rules import RuleResult, apply_rules, compute_candidates_proba
from nallm.postprocess.postprocess import PostProcessor

__all__ = [
    # 配置
    "THETA_A",
    "THETA_B",
    "MODEL_PATH",
    # 模型
    "DisambiguationModel",
    # 特征查询
    "FEATURE_DIM",
    "FeatureLookup",
    # 规则
    "RuleResult",
    "apply_rules",
    "compute_candidates_proba",
    # 主流程
    "PostProcessor",
]
