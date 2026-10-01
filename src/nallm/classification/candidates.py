"""候选作者工具模块

提供候选作者的排序、格式化等通用功能。

用法:
    from nallm.classification.candidates import (
        candidate_sort_key_v3,
        sort_candidates_v3,
        format_candidates,
    )
"""


def candidate_sort_key_v3(c: dict) -> tuple:
    """v3 排序键：强合作者 > 弱合作者 > 无合作者

    排序优先级:
    1. 是否强匹配 (coauthor_count > 0 AND coauthor_org_match_count > 0)
    2. coauthor_org_match_count 降序
    3. org_sim 降序
    4. venue_sim 降序
    5. title_sim_max 降序

    Args:
        c: 候选作者字典

    Returns:
        排序键元组
    """
    co = c.get("coauthor_count", 0)
    org = c.get("coauthor_org_match_count", 0)
    if co > 0 and org > 0:
        bucket = 0
    elif co > 0:
        bucket = 1
    else:
        bucket = 2
    return (
        bucket,
        -org,
        -c.get("org_sim", c.get("org_sim_max", c.get("main_org_sim", 0))),
        -c.get("venue_sim", c.get("venue_sim_max", 0)),
        -c.get("title_sim_max", c.get("title_sim", 0)),
    )


def sort_candidates_v3(candidates: list[dict]) -> list[dict]:
    """按 v3 排序键排序候选列表"""
    return sorted(candidates, key=candidate_sort_key_v3)


def format_candidates(sorted_cands: list[dict], max_show: int = 10) -> tuple[str, dict]:
    """格式化候选列表为人类可读字符串（用于 LLM prompt）

    Args:
        sorted_cands: 排序后的候选列表
        max_show: 最多展示数

    Returns:
        (格式化文本, aid_map)
        - aid_map: {"1": aid, "01": aid, ...} 编号到 aid 的映射
    """
    lines = ["候选作者列表（按可能性排序，编号仅用于引用）："]
    aid_map = {}
    for i, c in enumerate(sorted_cands[:max_show], 1):
        aid = c.get("aid") or c.get("candidate_aid", "?")
        sim = c.get("title_sim_max", 0)
        org_sim = c.get("org_sim", c.get("org_sim_max", 0))
        venue_sim = c.get("venue_sim", c.get("venue_sim_max", 0))
        co = c.get("coauthor_count", 0)
        org_match = c.get("coauthor_org_match_count", 0)
        lines.append(
            f"[{i}] aid={aid} | title_sim={sim:.3f} | org_sim={org_sim:.3f} | "
            f"venue_sim={venue_sim:.3f} | coauthor={co} | org_match={org_match}"
        )
        aid_map[str(i)] = aid
    if len(sorted_cands) > max_show:
        lines.append(f"... 共 {len(sorted_cands)} 个候选")
    return "\n".join(lines), aid_map


__all__ = [
    "candidate_sort_key_v3",
    "sort_candidates_v3",
    "format_candidates",
]
