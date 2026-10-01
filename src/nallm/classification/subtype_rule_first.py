"""子类型分类模块 — rule_first 体系（4+1 类，规则优先）

基于候选作者的合作者匹配强度和机构信号划分 4 个子类型 + NONE。

【分类体系 rule_first（扩展后）】

| 类别      | 触发条件                                                  | 策略       |
|-----------|----------------------------------------------------------|------------|
| AUTO      | 4 条 AUTO 规则（见 classify_rule_first）                 | AUTO       |
| LLM-Co    | co≥1 且非 AUTO/NONE                                       | SIMPLIFIED |
| LLM-NoCo  | co=0 AND org_sim≥0.85 且非 AUTO/NONE                     | P0_HIGH    |
| NONE      | 无候选 OR co=0 AND (org_sim<0.85 OR 三条件规则)           | NONE       |

【NONE 规则】
1. co=0 AND org_sim < 0.85（弱机构匹配）
2. co=0 AND sim<0.65 AND venue_sim<0.7 AND org_sim<0.95（极弱信号三无）
   命中率 97.06%，误判 3/102

【AUTO 规则（4 条）】
1. co≥2 AND org_match≥1（多合作者 + 机构验证）
2. co=0 AND org_sim≥0.95 AND sim≥0.70（强机构 + 中等标题）
3. org_match≥1 AND sim≥0.90（机构验证 + 极强标题）— 新增
4. org_match≥1 AND sim≥0.85 AND venue_sim≥0.85（机构验证 + 强标题 + 期刊匹配）— 新增

新规则 3+4 误分配仅 6 条（1.2%），高质量扩展。

【特征定义】
- co = coauthor_count（共同合作者匹配数）
- org_match = coauthor_org_match_count（机构验证通过的合作者数）
- sim = title_sim_max（候选最强论文与新论文的标题相似度）
- org_sim = 候选作者机构与新论文作者机构的余弦相似度
- venue_sim = 候选期刊与新论文期刊的余弦相似度

【统计（验证集）】
- 总案例: 13825
- NONE: 4487 (32.5%, 命中率 97.91%)
- AUTO: 5245 (37.9%, 命中率 97.21%)
- LLM-Co: 3659 (26.5%)
- LLM-NoCo: 434 (3.1%)
- LLM 压力降低 9.7%（vs 9 类版）

用法:
    from nallm.classification.subtype_rule_first import classify_rule_first

    subtype, details = classify_rule_first(candidates)
"""


# ============== 排序键 ==============
# 从 candidates 模块导入，保持向后兼容
from nallm.classification.candidates import candidate_sort_key_v3, sort_candidates_v3


# ============== NONE/AUTO 阈值常量 ==============
# NONE 规则
_NONE_WEAK_ORG_SIM = 0.85        # co=0 + org_sim<此值 → NONE
_NONE_TRIPLE_WEAK_SIM = 0.65     # 三无规则标题阈值
_NONE_TRIPLE_WEAK_VENUE = 0.70   # 三无规则期刊阈值
_NONE_TRIPLE_ORG = 0.95          # 三无规则机构阈值

# AUTO 规则
_AUTO_MIN_CO = 2                 # rule 1: 最少合作者匹配数
_AUTO_MIN_ORG_MATCH = 1          # rule 1/3/4: 最少机构验证数
_AUTO_ORG_SIM = 0.95             # rule 2: 强机构匹配
_AUTO_TITLE_WITH_ORG = 0.70      # rule 2: 配合强机构的标题阈值
_AUTO_STRONG_TITLE = 0.90        # rule 3: 极强标题
_AUTO_STRONG_TITLE_VENUE = 0.85  # rule 4: 强标题 + 强期刊
_AUTO_VENUE = 0.85               # rule 4: 期刊匹配


# ============== 主分类函数 ==============

def classify_rule_first(candidates: list[dict]) -> tuple[str, dict]:
    """简化版 4+1 分类

    Args:
        candidates: 候选作者列表（任意顺序，函数内部排序）

    Returns:
        (subtype, details) 元组:
        - subtype: "NONE" / "A1-Strong" (AUTO) / "LLM-Co" / "LLM-NoCo"
        - details: 包含决策依据的字典

    NONE 触发条件（2 条规则）:
        1. co=0 AND org_sim < 0.85（无合作者 + 弱机构匹配）
        2. co=0 AND sim<0.65 AND venue_sim<0.7 AND org_sim<0.95
           （三无极弱信号：弱标题 + 弱期刊 + 中等机构，命中率 97.06%）

    AUTO 触发条件（4 条规则）:
        1. co≥2 AND org_match≥1（多合作者 + 机构验证）
        2. co=0 AND org_sim≥0.95 AND sim≥0.70（强机构 + 中等标题）
        3. org_match≥1 AND sim≥0.90（机构验证 + 极强标题）— 新增
        4. org_match≥1 AND sim≥0.85 AND venue_sim≥0.85（机构验证 + 强标题 + 期刊匹配）— 新增

    LLM-Co: co≥1 且非 AUTO/NONE
    LLM-NoCo: co=0 AND org_sim≥0.85 且非 AUTO/NONE
    """
    # 0. 无候选
    if not candidates:
        return "NONE", {"reason": "no_candidates"}

    # 1. 按新排序键排序
    sorted_candidates = sort_candidates_v3(candidates)
    top1 = sorted_candidates[0]
    top3 = sorted_candidates[:3]

    # 2. 提取 Top1 特征（兼容 valid(org_sim/venue_sim) 和 test(org_sim_max/venue_sim_max)）
    co = top1.get("coauthor_count", 0)
    org_match = top1.get("coauthor_org_match_count", 0)
    sim = top1.get("title_sim_max", top1.get("title_sim", 0))
    org_sim = top1.get("org_sim", top1.get("org_sim_max", top1.get("main_org_sim", 0)))
    venue_sim = top1.get("venue_sim", top1.get("venue_sim_max", 0))

    # 3. NONE 硬规则（2 条）：
    if co == 0 and org_sim < _NONE_WEAK_ORG_SIM:
        return "NONE", {
            "reason": "no_coauthor_weak_org_match",
            "top1_coauthor": 0,
            "top1_title_sim": round(sim, 4),
            "top1_org_sim": round(org_sim, 4),
        }
    if co == 0 and sim < _NONE_TRIPLE_WEAK_SIM and venue_sim < _NONE_TRIPLE_WEAK_VENUE and org_sim < _NONE_TRIPLE_ORG:
        return "NONE", {
            "reason": "weak_title_weak_venue_medium_org",
            "top1_coauthor": 0,
            "top1_title_sim": round(sim, 4),
            "top1_org_sim": round(org_sim, 4),
            "top1_venue_sim": round(venue_sim, 4),
        }

    # 4. AUTO 规则（4 条）：
    if co >= _AUTO_MIN_CO and org_match >= _AUTO_MIN_ORG_MATCH:
        return "AUTO", {
            "reason": "multi_coauthor_org_verified",
            "top1_coauthor": co,
            "top1_org_match": org_match,
        }
    if co == 0 and org_sim >= _AUTO_ORG_SIM and sim >= _AUTO_TITLE_WITH_ORG:
        return "AUTO", {
            "reason": "strong_org_match",
            "top1_title_sim": round(sim, 4),
            "top1_org_sim": round(org_sim, 4),
        }
    if org_match >= _AUTO_MIN_ORG_MATCH and sim >= _AUTO_STRONG_TITLE:
        return "AUTO", {
            "reason": "org_verified_strong_title",
            "top1_coauthor": co,
            "top1_org_match": org_match,
            "top1_title_sim": round(sim, 4),
        }
    if org_match >= _AUTO_MIN_ORG_MATCH and sim >= _AUTO_STRONG_TITLE_VENUE and venue_sim >= _AUTO_VENUE:
        return "AUTO", {
            "reason": "org_verified_strong_title_venue",
            "top1_coauthor": co,
            "top1_org_match": org_match,
            "top1_title_sim": round(sim, 4),
            "top1_venue_sim": round(venue_sim, 4),
        }

    # 5. LLM-Co / LLM-NoCo 划分
    if co >= 1:
        subtype = "LLM-Co"
    else:
        subtype = "LLM-NoCo"

    # 6. 构建详情
    strong_count = sum(
        1 for c in top3
        if c.get("coauthor_count", 0) > 0
        and c.get("coauthor_org_match_count", 0) > 0
    )
    details = {
        "subtype": subtype,
        "top1_coauthor": co,
        "top1_org_match": org_match,
        "top1_title_sim": round(sim, 4),
        "top1_org_sim": round(org_sim, 4),
        "top1_venue_sim": round(venue_sim, 4),
        "strong_count": strong_count,
    }

    return subtype, details


# ============== 向后兼容 ==============

# llm_first 子类型 → rule_first 子类型映射
LLM_FIRST_TO_RULE_FIRST_MAP = {
    # AUTO
    "P1-Strong": "AUTO",
    "A1-Strong": "AUTO",
    # LLM-Co（有合作者匹配）
    "P1-High": "LLM-Co",
    "P1-Medium": "LLM-Co",
    "P1-Weak": "LLM-Co",
    "P1-Strong-Comp": "LLM-Co",
    "P1-High-Comp": "LLM-Co",
    "P1-Medium-Comp": "LLM-Co",
    "P1-Weak-Comp": "LLM-Co",
    "P2": "LLM-Co",
    "P2-Comp": "LLM-Co",
    "P3": "LLM-Co",
    "P3-Comp": "LLM-Co",
    "A2-Confirm": "LLM-Co",
    "A2-Weak-Co": "LLM-Co",
    "B-Comp-2way": "LLM-Co",
    "B-Comp-3way": "LLM-Co",
    # LLM-NoCo（无合作者匹配）
    "P0-High": "LLM-NoCo",
    "P0-Medium": "LLM-NoCo",
    "P0-Low": "LLM-NoCo",
    "C0-Org-Strong": "LLM-NoCo",
    "C0-Org-Medium": "LLM-NoCo",
    "C0-NoOrg": "LLM-NoCo",
    "C1-DefaultNone": "LLM-NoCo",
    # NONE
    "NONE": "NONE",
}


def legacy_to_rule_first_subtype(legacy: str) -> str:
    """llm_first → rule_first 子类型映射（仅用于日志/分析）"""
    return LLM_FIRST_TO_RULE_FIRST_MAP.get(legacy, "LLM-Co")


__all__ = [
    "candidate_sort_key_v3",
    "sort_candidates_v3",
    "classify_rule_first",
    "legacy_to_rule_first_subtype",
    "LLM_FIRST_TO_RULE_FIRST_MAP",
]
