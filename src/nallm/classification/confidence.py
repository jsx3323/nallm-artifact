"""置信度计算和校准模块

根据量化特征计算和校准消歧结果的置信度。
"""

# 置信度判断阈值
CONFIDENCE_ORG_SIM_HIGH = 0.7  # 机构相似度高置信度阈值
CONFIDENCE_ORG_SIM_STRONG = 0.8  # 机构相似度强阈值
CONFIDENCE_ORG_SIM_P0_MEDIUM = 0.85  # P0-Medium 机构相似度阈值
CONFIDENCE_VENUE_SIM_HIGH = 0.8  # 期刊相似度高置信度阈值
CONFIDENCE_VENUE_SIM_EXEMPTION = 0.9  # 期刊高度一致豁免阈值
CONFIDENCE_ORG_SIM_NO_MATCH = 0.5  # 机构完全不匹配阈值
CONFIDENCE_TITLE_SIM_STRONG = 0.9  # 标题相似度强阈值
CONFIDENCE_COAUTHOR_MIN = 1  # 最小合作者匹配数
CONFIDENCE_COAUTHOR_HIGH = 2  # 高置信度合作者匹配数
CONFIDENCE_COAUTHOR_EXCEPTION = 3  # 合作者豁免阈值（无需机构验证）

# 置信度排序
CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def compute_confidence(
    title_sim: float,
    coauthor_count: int,
    org_sim: float,
    venue_sim: float,
    org_verified: bool = False,
    predicted_author_id: str = "none",
) -> str:
    """根据特征量化计算置信度

    置信度判断规则：
    - high: 高可信合作者 >= 1 + 机构相似度 >= 0.7
    - high: 合作者 >= 2 + 机构相似度 >= 0.7 + 期刊相似度 >= 0.8
    - high: 标题相似度 >= 0.9 + 机构相似度 >= 0.8
    - medium: 高可信合作者 >= 1 (无机构验证)
    - medium: 合作者 >= 1 + 期刊相似度 >= 0.8
    - low: 其他情况

    Args:
        title_sim: 标题相似度
        coauthor_count: 合作者匹配数量
        org_sim: 机构相似度
        venue_sim: 期刊相似度
        org_verified: 是否有机构验证通过的合作者
        predicted_author_id: 预测的作者ID（用于判断 none）

    Returns:
        置信度字符串: "high", "medium", "low"
    """
    # 如果预测为 none，置信度为 low
    if predicted_author_id == "none":
        return "low"

    # High 置信度条件
    # 条件1: 高可信合作者匹配 + 机构匹配
    if org_verified and org_sim >= CONFIDENCE_ORG_SIM_HIGH:
        return "high"

    # 条件2: 合作者 >= 2 + 机构匹配 + 期刊匹配
    if (
        coauthor_count >= CONFIDENCE_COAUTHOR_HIGH
        and org_sim >= CONFIDENCE_ORG_SIM_HIGH
        and venue_sim >= CONFIDENCE_VENUE_SIM_HIGH
    ):
        return "high"

    # 条件3: 标题高度相似 + 机构强匹配
    if (
        title_sim >= CONFIDENCE_TITLE_SIM_STRONG
        and org_sim >= CONFIDENCE_ORG_SIM_STRONG
    ):
        return "high"

    # Medium 置信度条件
    # 条件1: 有合作者匹配但无机构验证
    if coauthor_count >= CONFIDENCE_COAUTHOR_MIN:
        return "medium"

    # 条件2: 有期刊匹配
    if venue_sim >= CONFIDENCE_VENUE_SIM_HIGH:
        return "medium"

    # 条件3: 有机构匹配
    if org_sim >= CONFIDENCE_ORG_SIM_HIGH:
        return "medium"

    # 默认 Low 置信度
    return "low"


def calibrate_confidence(
    llm_confidence: str,
    title_sim: float,
    coauthor_count: int,
    org_sim: float,
    venue_sim: float,
    org_verified: bool = False,
    predicted_author_id: str = "none",
    subtype: str = "",
) -> str:
    """校准 LLM 输出的置信度

    根据量化特征对 LLM 输出的置信度进行校准，防止过度乐观。

    Args:
        llm_confidence: LLM 输出的置信度
        title_sim: 标题相似度
        coauthor_count: 合作者匹配数量
        org_sim: 机构相似度
        venue_sim: 期刊相似度
        org_verified: 是否有机构验证通过的合作者
        predicted_author_id: 预测的作者ID
        subtype: 子类型（用于特殊规则）

    Returns:
        校准后的置信度
    """
    # 如果预测为 none，保持 LLM 的判断
    if predicted_author_id == "none":
        return llm_confidence

    # 计算量化置信度
    quantified = compute_confidence(
        title_sim=title_sim,
        coauthor_count=coauthor_count,
        org_sim=org_sim,
        venue_sim=venue_sim,
        org_verified=org_verified,
        predicted_author_id=predicted_author_id,
    )

    # 如果量化置信度低于 LLM 置信度，降级
    if CONFIDENCE_ORDER.get(quantified, 1) < CONFIDENCE_ORDER.get(llm_confidence, 1):
        return quantified

    # 特殊情况：机构相似度很低时强制降级
    if llm_confidence == "high":
        if (
            org_sim < CONFIDENCE_ORG_SIM_NO_MATCH
            and coauthor_count < CONFIDENCE_COAUTHOR_HIGH
        ):
            return "medium"
        if (
            org_sim < CONFIDENCE_ORG_SIM_NO_MATCH
            and venue_sim < CONFIDENCE_VENUE_SIM_HIGH
        ):
            return "medium"

    # P2/P3 特殊规则：合作者匹配位置靠后，需要更严格的条件
    if subtype.startswith("P2") or subtype.startswith("P3"):
        if not org_verified and llm_confidence == "high":
            return "medium"
        if org_sim < CONFIDENCE_ORG_SIM_NO_MATCH:
            return "low"

    # P1-Weak 特殊规则：弱信号案例，需保守判断
    if subtype.startswith("P1-Weak"):
        if not org_verified and llm_confidence == "high":
            return "medium"
        if org_sim < CONFIDENCE_ORG_SIM_HIGH:
            return "low"
        if coauthor_count == 1 and org_sim < CONFIDENCE_TITLE_SIM_STRONG:
            return "low"

    # P1-High/P1-Medium 共用逻辑：期刊高度一致可豁免机构验证
    venue_high = venue_sim >= CONFIDENCE_VENUE_SIM_EXEMPTION
    if subtype in ("P1-High", "P1-Medium"):
        coauthor_threshold = (
            CONFIDENCE_COAUTHOR_HIGH
            if subtype == "P1-High"
            else CONFIDENCE_COAUTHOR_MIN
        )
        if not venue_high:
            if org_sim < CONFIDENCE_ORG_SIM_NO_MATCH:
                return "low"
            if coauthor_count == 1 and not org_verified:
                return "low"
        if (
            coauthor_count >= coauthor_threshold
            and not org_verified
            and org_sim < CONFIDENCE_ORG_SIM_HIGH
            and not venue_high
        ):
            return "medium"

    # P0-Medium 特殊规则：无合作者匹配，需多重验证
    if subtype == "P0-Medium":
        if org_sim < CONFIDENCE_ORG_SIM_P0_MEDIUM:
            return "low"
        if (
            venue_sim < CONFIDENCE_VENUE_SIM_HIGH
            and org_sim < CONFIDENCE_ORG_SIM_STRONG
        ):
            return "low"

    # P1-*-Comp 竞争规则（除 P1-Weak-Comp，已被上面的 P1-Weak 规则覆盖）
    if subtype in ["P1-Strong-Comp", "P1-High-Comp", "P1-Medium-Comp"]:
        high_coauthor_exception = (
            coauthor_count >= CONFIDENCE_COAUTHOR_EXCEPTION
            and venue_sim >= CONFIDENCE_VENUE_SIM_EXEMPTION
        )
        if not high_coauthor_exception:
            if not org_verified and llm_confidence == "high":
                return "medium"
            if org_sim < CONFIDENCE_ORG_SIM_NO_MATCH:
                return "low"
            if (
                coauthor_count < CONFIDENCE_COAUTHOR_HIGH
                and org_sim < CONFIDENCE_ORG_SIM_HIGH
            ):
                return "low"

    return llm_confidence


__all__ = [
    "compute_confidence",
    "calibrate_confidence",
    "CONFIDENCE_ORG_SIM_HIGH",
    "CONFIDENCE_ORG_SIM_STRONG",
    "CONFIDENCE_ORG_SIM_P0_MEDIUM",
    "CONFIDENCE_VENUE_SIM_HIGH",
    "CONFIDENCE_VENUE_SIM_EXEMPTION",
    "CONFIDENCE_ORG_SIM_NO_MATCH",
    "CONFIDENCE_TITLE_SIM_STRONG",
    "CONFIDENCE_COAUTHOR_MIN",
    "CONFIDENCE_COAUTHOR_HIGH",
    "CONFIDENCE_COAUTHOR_EXCEPTION",
    "CONFIDENCE_ORDER",
]
