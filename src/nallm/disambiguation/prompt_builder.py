"""消歧 Prompt 构建模块

提供候选作者信息构建、Prompt 文本生成等功能。
"""

import hashlib
from collections import Counter

import numpy as np

from dataclasses import dataclass

from nallm.core.config import (
    ORG_MAX_LENGTH,
    ORG_SIM_THRESHOLD,
    PUB_ORG_SIM_THRESHOLD,
    PUB_VENUE_SIM_THRESHOLD,
)
from nallm.data.similarity import aggregate_candidate_stats
from nallm.embedding.database import EmbeddingDB
from nallm.disambiguation.strategy import DisambiguationStrategy


MAX_PUBS_PER_CANDIDATE = 10


@dataclass
class PromptData:
    """Prompt 构建数据

    包含消歧所需的完整 prompt 信息。
    """

    system_prompt: str
    human_text: str
    invoke_params: dict
    competition_info: dict | None = None

    def estimate_tokens(self, tokenizer_fn) -> int:
        """计算预估 token 数"""
        return tokenizer_fn(self.system_prompt) + tokenizer_fn(self.human_text)


def create_org_embedding_getter(db: EmbeddingDB):
    """创建 org embedding 获取函数"""

    def get_org_embedding(org: str) -> np.ndarray | None:
        if not org or not org.strip():
            return None
        org_hash = hashlib.md5(org.strip()[:ORG_MAX_LENGTH].encode()).hexdigest()
        return db.get(org_hash)

    return get_org_embedding


def get_title_bucket(title_sim: float) -> int:
    """将 title_sim 映射到分桶

    Args:
        title_sim: 标题相似度

    Returns:
        桶编号（0-3，越小越相似）
    """
    if title_sim >= 0.9:
        return 0
    elif title_sim >= 0.8:
        return 1
    elif title_sim >= 0.7:
        return 2
    else:
        return 3


def compute_paper_sort_key(sim_data_pub: dict) -> tuple:
    """计算论文排序键

    排序优先级（从高到低）：
    Tier 0: 高可信合作者匹配（coauthor_count > 0 且 coauthor_org_match > 0）
    Tier 1: 低可信合作者匹配（coauthor_count > 0）
    Tier 2: 机构+期刊双重匹配
    Tier 3: 单项匹配（机构或期刊）
    Tier 4: 仅标题相似度

    同层级内: title_bucket -> year_diff -> 具体分数

    Returns:
        (tier, title_bucket, year_diff, -primary, -secondary, -tertiary)
    """
    coauthor_count = sim_data_pub.get("coauthor_count", 0)
    coauthor_org_match = sim_data_pub.get("coauthor_org_match_count", 0)
    org_sim = sim_data_pub.get("org_sim", 0.0)
    venue_sim = sim_data_pub.get("venue_sim", 0.0)
    title_sim = sim_data_pub.get("title_sim", 0.0)
    year_diff = sim_data_pub.get("year_diff", 999)

    org_match = org_sim >= PUB_ORG_SIM_THRESHOLD
    venue_match = venue_sim >= PUB_VENUE_SIM_THRESHOLD
    title_bucket = get_title_bucket(title_sim)

    if coauthor_count > 0 and coauthor_org_match > 0:
        tier, primary, secondary, tertiary = (
            0,
            coauthor_count,
            coauthor_org_match,
            title_sim,
        )
    elif coauthor_count > 0:
        tier, primary, secondary, tertiary = 1, coauthor_count, title_sim, venue_sim
    elif org_match and venue_match:
        tier, primary, secondary, tertiary = 2, org_sim, venue_sim, title_sim
    elif org_match or venue_match:
        tier, primary, secondary, tertiary = (
            3,
            max(org_sim, venue_sim),
            min(org_sim, venue_sim),
            title_sim,
        )
    else:
        tier, primary, secondary, tertiary = 4, title_sim, 0.0, 0.0

    return (tier, title_bucket, year_diff, -primary, -secondary, -tertiary)


def sort_papers_by_relevance(
    all_pubs: list[str],
    pub_sims: dict[str, dict],
) -> list[str]:
    """按相关性排序论文

    Args:
        all_pubs: 论文ID列表
        pub_sims: 论文相似度数据 {pub_id: sim_data}

    Returns:
        排序后的论文ID列表
    """

    def get_sort_key(pid: str) -> tuple:
        sim = pub_sims.get(pid, {})
        return compute_paper_sort_key(sim)

    return sorted(all_pubs, key=get_sort_key)


def extract_main_org(author_pubs: list[dict]) -> str:
    """从作者论文中提取主要机构"""
    orgs = []
    for pub in author_pubs:
        for author in pub.get("authors", []):
            org = author.get("org", "").strip()
            if org and len(org) > 3:
                orgs.append(org)

    if not orgs:
        return "N/A"

    counter = Counter(orgs)
    return counter.most_common(1)[0][0]


def extract_author_keywords(author_pubs: list[dict]) -> set[str]:
    """从作者论文中提取关键词集合"""
    keywords = set()
    for pub in author_pubs:
        pub_keywords = pub.get("keywords", [])
        if pub_keywords:
            keywords.update(kw.lower().strip() for kw in pub_keywords if kw)
    return keywords


def construct_candidate_info(
    unass_record: str,
    aid: str,
    author_info: dict,
    whole_author_profiles_pub: dict,
    sim_data: dict[str, dict[str, dict[str, dict]]],
    compute_org_similarity_fn,
    unass_keywords: set[str] | None = None,
    unass_org: str | None = None,
) -> str:
    """构造候选作者信息

    Args:
        unass_record: 待消歧记录 ID
        aid: 候选作者 ID
        author_info: 候选作者信息
        whole_author_profiles_pub: 所有论文信息字典
        sim_data: 相似度数据
        compute_org_similarity_fn: 机构相似度计算函数
        unass_keywords: 待消歧论文关键词
        unass_org: 待消歧作者机构

    Returns:
        格式化的候选作者信息字符串
    """
    all_pubs = author_info.get("pubs", [])
    total_pubs = len(all_pubs)

    # 获取相似度数据
    pub_sims = sim_data.get(unass_record, {}).get(aid, {})
    stats = aggregate_candidate_stats(pub_sims)

    # 按相关性排序论文
    sorted_pubs = sort_papers_by_relevance(all_pubs, pub_sims)
    selected_pubs = sorted_pubs[:MAX_PUBS_PER_CANDIDATE]

    # 获取论文详情
    pubs_detail = [
        whole_author_profiles_pub.get(pid, {})
        for pid in selected_pubs
        if pid in whole_author_profiles_pub
    ]

    main_org = extract_main_org(pubs_detail)

    # 计算关键词匹配
    author_keywords = extract_author_keywords(pubs_detail)
    keyword_match_count = 0
    matched_keywords = []
    if unass_keywords and author_keywords:
        matched_keywords = list(unass_keywords & author_keywords)
        keyword_match_count = len(matched_keywords)

    # 计算机构相似度
    org_sim = stats["max_org_sim"]
    if org_sim == 0 and unass_org and main_org and main_org != "N/A":
        org_sim = compute_org_similarity_fn(unass_org, main_org)

    if org_sim > 0:
        org_match_marker = "✓" if org_sim >= ORG_SIM_THRESHOLD else "?"
        org_display = f"{main_org} (相似度: {org_sim:.2f}{org_match_marker})"
    else:
        org_display = main_org

    coauthor_count = stats["total_coauthor_count"]
    coauthor_org_match = stats["coauthor_org_match_count"]

    # 为有合作者匹配的候选者添加醒目标记
    if coauthor_count > 0:
        author_header = f"【作者ID: {aid}】⚠️ 有合作者匹配"
    else:
        author_header = f"【作者ID: {aid}】"

    lines = [
        author_header,
        f"论文总数: {total_pubs}",
        f"主要机构: {org_display}",
        f"合作者匹配: {coauthor_count} (机构验证✓: {coauthor_org_match})",
        f"标题相似度: {stats['max_title_sim']:.2f}",
        f"机构相似度: {stats['max_org_sim']:.2f}",
        f"期刊相似度: {stats['max_venue_sim']:.2f}",
    ]

    if unass_keywords is not None:
        lines.append(f"关键词匹配数: {keyword_match_count}")
        if matched_keywords:
            lines.append(f"匹配关键词: {', '.join(matched_keywords[:10])}")

    lines.append(f"代表性论文（展示前 {MAX_PUBS_PER_CANDIDATE} 篇）:")

    for pid in selected_pubs:
        pub = whole_author_profiles_pub.get(pid, {})
        if not pub:
            continue

        sim_data_pub = pub_sims.get(pid, {})
        title_sim_val = sim_data_pub.get("title_sim", 0.0)
        abstract_sim_val = sim_data_pub.get("abstract_sim", 0.0)
        venue_sim_val = sim_data_pub.get("venue_sim", 0.0)

        lines.append(f"\n  - 论文ID: {pid}")
        lines.append(
            f"    标题: {pub.get('title', 'N/A')} (相似度: {title_sim_val:.2f})"
        )
        lines.append(f"    摘要相似度: {abstract_sim_val:.2f}")

        lines.append("    作者列表:")
        for author in pub.get("authors", [])[:10]:
            lines.append(
                f"      - {author.get('name', 'N/A')} ({author.get('org', 'N/A')})"
            )

        lines.append(
            f"    发表: {pub.get('venue', 'N/A')}, {pub.get('year', 'N/A')} (相似度: {venue_sim_val:.2f})"
        )

    return "\n".join(lines)


def construct_unass_pub_info(pub: dict, author_order: int) -> str:
    """构造待消歧论文信息

    Args:
        pub: 论文信息字典
        author_order: 待消歧作者的顺序（0-indexed）

    Returns:
        格式化的论文信息字符串
    """
    abstract = pub.get("abstract", "")
    abstract_display = abstract[:500] + "..." if len(abstract) > 500 else abstract

    authors = pub.get("authors", [])
    unass_author = authors[author_order] if author_order < len(authors) else {}
    unass_org = unass_author.get("org", "")

    keywords = pub.get("keywords", [])

    lines = [
        "【待消歧论文】",
        f"标题: {pub.get('title', 'N/A')}",
        f"摘要: {abstract_display}",
        f"发表: {pub.get('venue', 'N/A')}, {pub.get('year', 'N/A')}",
    ]

    if unass_org:
        lines.append(f"待消歧作者机构: {unass_org}")

    if keywords:
        lines.append(f"关键词: {', '.join(keywords[:10])}")

    lines.extend(
        [
            "",
            "完整作者列表（标记 * 的为待消歧作者）:",
        ]
    )

    for idx, author in enumerate(authors):
        marker = " *" if idx == author_order else ""
        org = author.get("org", "")
        lines.append(f"  {idx + 1}. {author.get('name', 'N/A')}{marker} - {org}")

    return "\n".join(lines)


def construct_prompt_data(
    unass_record: str,
    candidates: list[dict],
    data: dict,
    strategy: DisambiguationStrategy,
    max_candidates: int = 10,
    is_test: bool = False,
    sim_data: dict | None = None,
    compute_org_similarity_fn=None,
    get_system_prompt_fn=None,
    build_human_text_fn=None,
) -> PromptData | None:
    """构造消歧 Prompt 数据

    统一处理 prompt 构建逻辑，供 disambiguate_record 和 estimate_record_tokens 共用。

    Args:
        unass_record: 待消歧记录 ID
        candidates: 候选作者列表
        data: 数据字典
        strategy: 消歧策略
        max_candidates: 最大候选数
        is_test: 是否测试集
        sim_data: 相似度数据
        compute_org_similarity_fn: 机构相似度计算函数
        get_system_prompt_fn: 获取 system prompt 的函数
        build_human_text_fn: 构建 human text 的函数

    Returns:
        PromptData 实例，若记录格式无效返回 None
    """
    # 解析记录 ID
    try:
        unass_pid, unass_author_order = unass_record.split("-", 1)
        unass_author_order = int(unass_author_order)
    except ValueError:
        return None

    # 选择数据源
    unass_pub_data = data.get("test_unass_pub" if is_test else "valid_unass_pub", {})
    whole_author_profiles = data.get("whole_author_profiles", {})
    whole_author_profiles_pub = data.get("whole_author_profiles_pub", {})

    if unass_pid not in unass_pub_data:
        return None

    unass_pub_info = unass_pub_data[unass_pid]

    # 提取关键词和机构
    unass_keywords = set(
        kw.lower().strip() for kw in unass_pub_info.get("keywords", []) if kw
    )
    unass_authors = unass_pub_info.get("authors", [])
    unass_org = (
        unass_authors[unass_author_order].get("org", "")
        if unass_author_order < len(unass_authors)
        else ""
    )

    # 构造候选作者信息
    candidate_infos = []
    for cand in candidates[:max_candidates]:
        aid = cand["aid"]
        author_info = whole_author_profiles.get(aid, {})

        cand_info = construct_candidate_info(
            unass_record,
            aid,
            author_info,
            whole_author_profiles_pub,
            sim_data or {},
            compute_org_similarity_fn,
            unass_keywords,
            unass_org,
        )
        candidate_infos.append(cand_info)

    # 构造待消歧论文信息
    unass_pub_display = construct_unass_pub_info(unass_pub_info, unass_author_order)

    # 准备 invoke 参数
    invoke_params = {
        "candidates": "\n\n".join(candidate_infos),
        "new_pub": unass_pub_display,
    }

    # P0-High 添加竞争检测信息
    competition_info = None
    if strategy == DisambiguationStrategy.P0_HIGH:
        num_candidates = len(candidates)
        top1_title_sim = candidates[0]["title_sim"] if candidates else 0.0
        top2_title_sim = candidates[1]["title_sim"] if len(candidates) > 1 else 0.0
        title_sim_gap = top1_title_sim - top2_title_sim
        has_competition = num_candidates > 10 or title_sim_gap < 0.05

        competition_info = {
            "num_candidates": num_candidates,
            "top1_title_sim": f"{top1_title_sim:.3f}",
            "top2_title_sim": f"{top2_title_sim:.3f}",
            "title_sim_gap": f"{title_sim_gap:.3f}",
            "has_competition": "是" if has_competition else "否",
        }

        invoke_params.update(competition_info)

    # 获取 system prompt
    system_prompt = get_system_prompt_fn(strategy) if get_system_prompt_fn else ""

    # 构建 human text
    human_text = (
        build_human_text_fn(
            strategy=strategy,
            candidates_text=invoke_params["candidates"],
            new_pub_text=invoke_params["new_pub"],
            competition_info=competition_info,
        )
        if build_human_text_fn
        else ""
    )

    return PromptData(
        system_prompt=system_prompt,
        human_text=human_text,
        invoke_params=invoke_params,
        competition_info=competition_info,
    )


def build_llm_co_prompt(
    new_pub: dict,
    samples_by_aid: dict,
    candidates: list[dict],
    author_order: int = 0,
) -> str:
    """构造 LLM-Co prompt 文本（合 5 类：P1-High/Strong-Comp/High-Comp/Medium/Weak 系列 + P2/P3）

    Args:
        new_pub: 待消歧论文（含 title/authors/year/venue）
        samples_by_aid: {aid: [PaperSample, ...]} 已采样的论文
        candidates: [{aid, name, coauthor_count, ...}, ...] 候选作者汇总
        author_order: 待消歧作者在 authors 中的位置

    Returns:
        完整 prompt 文本
    """
    new_authors = new_pub.get("authors", [])
    new_author = new_authors[author_order] if author_order < len(new_authors) else {}
    new_org = new_author.get("org", "")
    new_year = new_pub.get("year", "")
    new_venue = new_pub.get("venue", "")
    new_kw = new_pub.get("keywords", []) or []

    lines: list[str] = []
    lines.append("# 待消歧论文")
    lines.append(f"标题：{new_pub.get('title', 'N/A')}")
    lines.append(f"作者：{new_author.get('name', 'N/A')}")
    if new_org:
        lines.append(f"机构：{new_org}")
    if new_year:
        lines.append(f"年份：{new_year}")
    if new_venue:
        lines.append(f"发表：{new_venue}")
    if new_kw:
        lines.append(f"关键词：{', '.join(str(k) for k in new_kw[:8])}")

    abstract = new_pub.get("abstract", "")
    if abstract:
        lines.append(f"\n摘要：{abstract[:400]}{'...' if len(abstract) > 400 else ''}")

    lines.append(f"\n完整作者列表（共 {len(new_authors)} 人）：")
    for i, a in enumerate(new_authors[:15]):
        marker = " *" if i == author_order else ""
        lines.append(f"  {i+1}. {a.get('name','?')}{marker} - {a.get('org','')}")
    if len(new_authors) > 15:
        lines.append(f"  ... (其余 {len(new_authors)-15} 人省略)")

    lines.append("\n# 候选作者")
    for cand in candidates:
        aid = cand.get("aid") or cand.get("candidate_aid") or cand.get("id")
        name = cand.get("name", "?")
        co_total = cand.get("coauthor_count", 0)
        org_match = cand.get("coauthor_org_match_count", 0)
        title_max = cand.get("title_sim_max") or cand.get("title_sim", 0.0)
        org_max = cand.get("org_sim", 0.0)
        venue_max = cand.get("venue_sim", 0.0)

        marker = " ⚠️有合作者匹配" if co_total > 0 else ""
        lines.append(f"\n## 候选 [{aid}] {name}{marker}")
        lines.append(
            f"合作者匹配：{co_total} 篇（机构验证 {org_match} 篇） | "
            f"title_sim={title_max:.2f} org_sim={org_max:.2f} venue_sim={venue_max:.2f}"
        )

        samples = samples_by_aid.get(aid, [])
        if not samples:
            lines.append("  （无可用论文）")
            continue
        lines.append(f"代表性论文（Top-{len(samples)}）：")
        for i, s in enumerate(samples, 1):
            p = s.pub
            title = (p.get("title") or "")[:100]
            year = p.get("year", "?")
            venue = p.get("venue", "")
            co_authors = []
            for a in p.get("authors", []):
                an = a.get("name")
                if an and an.lower() != name.lower():
                    co_authors.append(an)
                    if len(co_authors) == 6:
                        break
            lines.append(f"  [{i}] {year} - {title}")
            tag = f"co={s.co} ts={s.title_sim:.2f} vs={s.venue_sim:.2f} os={s.org_sim:.2f} yr={s.year_diff}"
            lines.append(f"      [{tag}] score={s.score:.0f}")
            if venue:
                lines.append(f"      发表：{venue}")
            if co_authors:
                lines.append(f"      合作者：{', '.join(co_authors)}")

    lines.append("\n# 任务")
    lines.append("判断待消歧论文的真实作者归属：")
    lines.append("")
    lines.append("【决策优先级】")
    lines.append("1. 若 co≥2 且 org_match≥1（机构验证通过）→ 极可能是真实作者")
    lines.append("2. 若 co≥2 但 org_match=0 → 强合作者信号可能为误报（同名作者），需结合 org_sim 判断")
    lines.append("   - 若同时 org_sim=0（机构完全不同）→ 强烈倾向 none")
    lines.append("   - 若同时 org_sim≥0.8（机构一致）→ 可信，仍选该候选")
    lines.append("3. 若 co=0 → 比较主题/机构/期刊相似度")
    lines.append("4. 若所有候选均为「无可用论文」或相似度都低 → 输出 none")
    lines.append("")
    lines.append("【关键反向信号】")
    lines.append("- org_sim=0（机构完全不同）+ co<3 → 倾向 none")
    lines.append("- 候选无可用论文 + co<2 → 倾向 none")
    lines.append("- ⚠️ 宁可输出 none 也不要在低质量候选中猜错")
    lines.append("")
    lines.append("【示例】")
    lines.append("例 1：co=2 org_match=0 ts=0.88 org_sim=0.00 → 输出 none（强 co 但机构完全不同）")
    lines.append("例 2：co=3 org_match=1 ts=0.85 org_sim=0.92 → 输出 top1 aid（co + 机构验证）")
    lines.append("例 3：co=0 ts=0.65 os=0.70 vs=0.75 → 输出 none（所有指标中等偏弱）")
    lines.append("")
    lines.append("【输出格式】")
    lines.append("answer: <候选 aid 或 none>")
    lines.append("confidence: <high/medium/low>")
    lines.append("reason: <简短理由>")

    return "\n".join(lines)


__all__ = [
    "PromptData",
    "construct_prompt_data",
    "create_org_embedding_getter",
    "get_title_bucket",
    "compute_paper_sort_key",
    "sort_papers_by_relevance",
    "extract_main_org",
    "extract_author_keywords",
    "construct_candidate_info",
    "construct_unass_pub_info",
    "build_llm_co_prompt",
    "MAX_PUBS_PER_CANDIDATE",
]
