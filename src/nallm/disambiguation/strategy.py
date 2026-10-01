"""消歧策略定义

定义消歧策略枚举和子类型到策略的映射逻辑。

用法:
    from nallm.disambiguation.strategy import DisambiguationStrategy, get_strategy

    strategy = get_strategy("AUTO")  # 返回 DisambiguationStrategy.AUTO
"""

from enum import Enum


class DisambiguationStrategy(Enum):
    """消歧策略枚举

    每种策略对应不同的消歧处理方式：
    - AUTO: 自动分配，无需 LLM（高置信度）
    - SIMPLIFIED: 简化 LLM prompt（高置信度）
    - STANDARD: 标准 LLM prompt
    - P0_*: P0 系列专用（无合作者匹配）
    - P1_*: P1 系列专用（有合作者匹配）
    - NONE: 直接返回 none（无候选）
    """

    AUTO = "auto"  # 自动分配，无需 LLM
    SIMPLIFIED = "simplified"  # 简化 LLM prompt
    STANDARD = "standard"  # 标准 LLM prompt
    P0_HIGH = "p0_high"  # P0-High 专用（带竞争检测）
    P0_MEDIUM = "p0_medium"  # P0-Medium 专用（谨慎判断）
    P0_LOW = "p0_low"  # P0-Low 专用（默认拒绝）
    P2_P3 = "p2_p3"  # P2/P3 专用（合作者匹配位置靠后）
    P1_WEAK = "p1_weak"  # P1-Weak 专用（弱信号，保守判断）
    P1_COMP = "p1_comp"  # 旧 P1-*-Comp 通用策略（向后兼容）
    P1_HIGH = "p1_high"  # P1-High 专用（合作者匹配 + 机构验证）
    P1_MEDIUM = "p1_medium"  # P1-Medium 专用（弱合作者匹配场景）
    NONE = "none"  # 直接返回 none


# ============== 子类型到策略映射 ==============

# rule_first 子类型 → 策略（首选）
RULE_FIRST_STRATEGY_MAP = {
    "NONE": DisambiguationStrategy.NONE,
    "AUTO": DisambiguationStrategy.AUTO,
    "LLM-Co": DisambiguationStrategy.SIMPLIFIED,
    "LLM-NoCo": DisambiguationStrategy.P0_HIGH,
}

# llm_first 子类型 → 策略
LLM_FIRST_STRATEGY_MAP = {
    # NONE: 无候选
    "NONE": DisambiguationStrategy.NONE,
    # AUTO: 自动分配，高置信度
    "P1-Strong": DisambiguationStrategy.AUTO,
    # P1-High: 合作者匹配信号强，需要机构验证
    "P1-High": DisambiguationStrategy.P1_HIGH,
    # P1-Medium: 合作者匹配信号中等，需要更严格验证
    "P1-Medium": DisambiguationStrategy.P1_MEDIUM,
    # P1-*-Comp: 竞争场景，使用专用策略
    "P1-Strong-Comp": DisambiguationStrategy.P1_COMP,
    "P1-High-Comp": DisambiguationStrategy.P1_COMP,
    "P1-Medium-Comp": DisambiguationStrategy.P1_COMP,
    # P1-Weak: 弱信号，使用专用保守策略
    "P1-Weak": DisambiguationStrategy.P1_WEAK,
    "P1-Weak-Comp": DisambiguationStrategy.P1_WEAK,
    # P2/P3: 合作者匹配位置靠后，使用专用策略
    "P2": DisambiguationStrategy.P2_P3,
    "P2-Comp": DisambiguationStrategy.P2_P3,
    "P3": DisambiguationStrategy.P2_P3,
    "P3-Comp": DisambiguationStrategy.P2_P3,
    # P0 系列：无合作者匹配，使用专用策略
    "P0-High": DisambiguationStrategy.P0_HIGH,
    "P0-Medium": DisambiguationStrategy.P0_MEDIUM,
    "P0-Low": DisambiguationStrategy.P0_LOW,
}

# 合并：rule_first 优先，回退到 llm_first
SUBTYPE_STRATEGY_MAP = {**LLM_FIRST_STRATEGY_MAP, **RULE_FIRST_STRATEGY_MAP}


def get_strategy(subtype: str) -> DisambiguationStrategy:
    """获取消歧策略

    Args:
        subtype: 子类型字符串
                rule_first 体系: "AUTO" / "LLM-Co" / "LLM-NoCo" / "NONE"
                llm_first 体系: "P1-Strong" / "P1-High" / "P0-High" / "P2" 等

    Returns:
        对应的消歧策略，未知子类型默认返回 SIMPLIFIED
    """
    return SUBTYPE_STRATEGY_MAP.get(subtype, DisambiguationStrategy.SIMPLIFIED)


__all__ = [
    "DisambiguationStrategy",
    "SUBTYPE_STRATEGY_MAP",
    "RULE_FIRST_STRATEGY_MAP",
    "LLM_FIRST_STRATEGY_MAP",
    "get_strategy",
]