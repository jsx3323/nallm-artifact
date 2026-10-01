"""子类型分类模块 — llm_first 体系（9 类细分，LLM 优先）

提供统一的子类型划分逻辑，适用于所有候选作者记录。

子类型结构：
1. P1/P2/P3: 按合作者匹配位置
   - P1: Top1 有合作者匹配，细分强度 (Strong/High/Medium/Weak)
   - P2: Top2 有合作者匹配 (Top1 无)
   - P3: Top3 有合作者匹配 (Top1/2 无)

2. P0: 无合作者匹配，按相似度细分
   - P0-High: title_sim >= 0.85
   - P0-Medium: 0.75 <= title_sim < 0.85
   - P0-Low: title_sim < 0.75

3. NONE: 无候选作者

4. 竞争标记 (-Comp): Top3 中有 >= 2 个候选有合作者匹配

用法:
    from nallm.classification.subtype_llm_first import classify_llm_first

    subtype, details = classify_llm_first(candidates)
"""

from nallm.core.config import (
    ORG_SIM_THRESHOLD,
    P0_HIGH_ORG_SIM,
    P0_HIGH_TITLE_SIM,
    P0_HIGH_VENUE_SIM,
    P0_MEDIUM_TITLE_SIM,
    P0_MEDIUM_TOP5_AVG,
    P1_HIGH_COAUTHOR,
    P1_HIGH_TITLE_SIM,
    P1_MEDIUM_TITLE_SIM,
    P1_STRONG_COAUTHOR,
    P1_STRONG_TITLE_SIM,
)
from nallm.name import normalize_name_key
from nallm.core.similarity import compute_org_similarity


def get_matched_coauthors_with_org(
    unass_pub_info: dict,
    unass_author_order: int,
    candidate_pubs: list[str],
    whole_author_profiles_pub: dict,
    org_sim_threshold: float = ORG_SIM_THRESHOLD,
) -> tuple[int, int, list[dict]]:
    """获取候选作者与待消歧论文的合作者匹配信息（带机构验证）

    Args:
        unass_pub_info: 待消歧论文信息
        unass_author_order: 待消歧作者的顺序（0-indexed）
        candidate_pubs: 候选作者的论文ID列表
        whole_author_profiles_pub: 所有论文信息字典
        org_sim_threshold: 机构相似度阈值（默认 0.7）

    Returns:
        (总匹配数量, 机构验证通过数量, 匹配详情列表)
    """
    unass_authors = unass_pub_info.get("authors", [])
    unass_coauthor_info: dict[str, list[str]] = {}

    for idx, author in enumerate(unass_authors):
        if idx != unass_author_order:
            name = author.get("name", "")
            org = author.get("org", "")
            if name:
                name_key = normalize_name_key(name)
                if name_key not in unass_coauthor_info:
                    unass_coauthor_info[name_key] = []
                if org:
                    unass_coauthor_info[name_key].append(org)

    matched_details: dict[str, dict] = {}

    for pub_id in candidate_pubs:
        if pub_id not in whole_author_profiles_pub:
            continue
        cand_authors = whole_author_profiles_pub[pub_id].get("authors", [])
        for author in cand_authors:
            name = author.get("name", "")
            org = author.get("org", "")
            if not name:
                continue

            name_key = normalize_name_key(name)
            if name_key not in unass_coauthor_info:
                continue

            if name not in matched_details:
                matched_details[name] = {
                    "name": name,
                    "org_match": False,
                    "max_org_sim": 0.0,
                }

            if org and unass_coauthor_info[name_key]:
                for unass_org in unass_coauthor_info[name_key]:
                    sim = compute_org_similarity(org, unass_org)
                    if sim > matched_details[name]["max_org_sim"]:
                        matched_details[name]["max_org_sim"] = sim
                    if sim >= org_sim_threshold:
                        matched_details[name]["org_match"] = True

    matched_list = list(matched_details.values())
    total_matched = len(matched_list)
    org_verified = sum(1 for d in matched_list if d["org_match"])

    return total_matched, org_verified, matched_list


def classify_llm_first(candidates: list[dict]) -> tuple[str, dict]:
    """统一子类型分类

    子类型结构：
    - NONE: 无候选作者
    - P1-Strong/High/Medium/Weak[-Comp]: Top1 有合作者匹配
    - P2[-Comp]: Top2 有合作者匹配
    - P3[-Comp]: Top3 有合作者匹配
    - P0-High/Medium/Low: 无合作者匹配，按相似度细分

    Args:
        candidates: 候选作者列表（已按 title_sim 降序排列）

    Returns:
        (子类型字符串, 详细信息字典)
    """
    if not candidates:
        return "NONE", {}

    # 1. 确定合作者匹配位置
    match_position = -1
    for i, cand in enumerate(candidates[:3], start=1):
        if cand.get("coauthor_count", 0) > 0:
            match_position = i
            break

    # 2. 获取 Top1 信息
    top1 = candidates[0]
    top1_title_sim = top1.get("title_sim", 0)

    # 3. 无合作者匹配的情况 (P0)
    if match_position == -1:
        title_sim = top1_title_sim
        # 获取多个相似度指标
        title_sim_max = top1.get("title_sim_max", top1.get("title_sim", 0))
        top5_avg = top1.get("title_sim_top5_avg", title_sim_max)
        org_sim = top1.get("org_sim", 0)
        venue_sim = top1.get("venue_sim", 0)

        # P0-High: 三重条件 (title_sim + org_sim + venue_sim)
        if (
            title_sim_max >= P0_HIGH_TITLE_SIM
            and org_sim >= P0_HIGH_ORG_SIM
            and venue_sim >= P0_HIGH_VENUE_SIM
        ):
            subtype = "P0-High"
        elif title_sim_max >= P0_MEDIUM_TITLE_SIM and top5_avg >= P0_MEDIUM_TOP5_AVG:
            subtype = "P0-Medium"
        else:
            subtype = "P0-Low"

        details = {
            "position": 0,
            "strength": subtype.split("-")[1] if "-" in subtype else None,
            "has_coauthor": False,
            "has_competition": False,
            "num_competitors": 0,
            "title_sim": round(title_sim_max, 4),
            "title_sim_max": round(title_sim_max, 4),
            "title_sim_top5_avg": round(top5_avg, 4),
            "org_sim": round(org_sim, 4),
            "venue_sim": round(venue_sim, 4),
            "coauthor_count": 0,
            "org_verified": False,
        }
        return subtype, details

    # 4. 有合作者匹配的情况 (P1/P2/P3)
    # 获取有合作者匹配的候选者信息
    if match_position > 0 and match_position <= len(candidates):
        matched_cand = candidates[match_position - 1]
        title_sim = matched_cand.get("title_sim", 0)
        coauthor_count = matched_cand.get("coauthor_count", 0)
        org_verified = matched_cand.get("coauthor_org_match_count", 0) > 0
    else:
        title_sim = top1_title_sim
        coauthor_count = 0
        org_verified = False

    # 计算竞争情况
    num_with_coauthor = sum(1 for c in candidates[:3] if c.get("coauthor_count", 0) > 0)
    has_competition = num_with_coauthor >= 2

    # 构建子类型
    strength = None
    if match_position == 1:
        # P1: 按强度细分
        if (
            title_sim >= P1_STRONG_TITLE_SIM
            and coauthor_count >= P1_STRONG_COAUTHOR
            and org_verified
        ):
            strength = "Strong"
        elif title_sim >= P1_HIGH_TITLE_SIM and coauthor_count >= P1_HIGH_COAUTHOR:
            strength = "High"
        elif title_sim >= P1_MEDIUM_TITLE_SIM:
            strength = "Medium"
        else:
            strength = "Weak"

        subtype = f"P1-{strength}"
    else:
        # P2/P3: 不细分强度
        subtype = f"P{match_position}"

    # 添加竞争标记
    if has_competition:
        subtype = f"{subtype}-Comp"

    # 返回详细信息
    details = {
        "position": match_position,
        "strength": strength,
        "has_coauthor": True,
        "has_competition": has_competition,
        "num_competitors": num_with_coauthor,
        "title_sim": round(title_sim, 4),
        "coauthor_count": coauthor_count,
        "org_verified": org_verified,
    }

    return subtype, details


# 向后兼容的别名
classify_l1_detailed = classify_llm_first


__all__ = [
    "get_matched_coauthors_with_org",
    "classify_llm_first",
    "classify_l1_detailed",
]
