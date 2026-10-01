"""后处理规则逻辑

实现规则 A（纠正误判）和规则 B（纠正漏判）。
"""

from dataclasses import dataclass

import numpy as np

from nallm.postprocess.config import THETA_A, THETA_B
from nallm.postprocess.feature_compute import FeatureLookup
from nallm.postprocess.model import DisambiguationModel

NONE_AUTHOR_ID = "none"


@dataclass
class RuleResult:
    """规则应用结果"""

    action: str  # "keep" | "override_to_none" | "override_to_candidate"
    original_aid: str  # LLM 原始预测的作者 ID
    modified_aid: str  # 修正后的作者 ID（与 original_aid 相同则未修改）
    model_proba: float  # 模型对原始预测候选的概率
    rule_applied: str | None  # "A" | "B" | None


def apply_rules(
    predicted_author_id: str,
    candidates_with_proba: list[tuple[str, float]],
    theta_a: float = THETA_A,
    theta_b: float = THETA_B,
) -> RuleResult:
    """对单条记录应用后处理规则

    Args:
        predicted_author_id: LLM 预测的作者 ID（"none" 表示未分配）
        candidates_with_proba: [(aid, model_proba), ...] 候选列表（按 proba 降序）
        theta_a: 规则 A 阈值
        theta_b: 规则 B 阈值

    Returns:
        RuleResult 规则应用结果
    """
    proba_map = {aid: proba for aid, proba in candidates_with_proba}

    # LLM 预测的 aid 不在候选列表中时，跳过规则 A
    llm_proba = proba_map.get(predicted_author_id, 0.0)

    # 规则 A: LLM 分配了 + 模型概率 < theta_a → 改为 none
    if predicted_author_id != NONE_AUTHOR_ID and predicted_author_id in proba_map:
        if llm_proba < theta_a:
            return RuleResult(
                action="override_to_none",
                original_aid=predicted_author_id,
                modified_aid=NONE_AUTHOR_ID,
                model_proba=llm_proba,
                rule_applied="A",
            )

    # 规则 B: LLM 返回 none + 模型 top1 概率 >= theta_b → 分配
    if predicted_author_id == NONE_AUTHOR_ID and candidates_with_proba:
        top1_aid, top1_proba = candidates_with_proba[0]
        if top1_proba >= theta_b:
            return RuleResult(
                action="override_to_candidate",
                original_aid=NONE_AUTHOR_ID,
                modified_aid=top1_aid,
                model_proba=top1_proba,
                rule_applied="B",
            )

    # 未触发任何规则
    return RuleResult(
        action="keep",
        original_aid=predicted_author_id,
        modified_aid=predicted_author_id,
        model_proba=llm_proba,
        rule_applied=None,
    )


def compute_candidates_proba(
    unass_record: str,
    feature_lookup: FeatureLookup,
    model: DisambiguationModel,
) -> list[tuple[str, float]]:
    """计算一条记录所有候选的模型概率

    Args:
        unass_record: 待消歧记录 ID
        feature_lookup: 特征查表器
        model: 消歧模型

    Returns:
        [(aid, proba), ...] 按 proba 降序排列
    """
    aid_features = feature_lookup.get_candidates_features(unass_record)

    if not aid_features:
        return []

    aids = [aid for aid, _ in aid_features]
    features = np.vstack([feat for _, feat in aid_features])
    probas = model.predict_proba(features)

    results = list(zip(aids, probas.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)

    return results


__all__ = [
    "RuleResult",
    "apply_rules",
    "compute_candidates_proba",
    "NONE_AUTHOR_ID",
]
