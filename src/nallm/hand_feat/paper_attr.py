"""论文属性提取模块

从论文信息中提取特征计算所需的属性：
- 合作者集合
- 机构
- 期刊/会议
- 关键词
- 标题
"""

from nallm.name import normalize_name, normalize_profile_name, name_match


def get_author_index(name: str, authors: list[dict]) -> int:
    """获取目标作者在作者列表中的索引

    Args:
        name: 目标作者姓名（下划线分隔格式，如 "wang_wei"）
        authors: 论文作者列表

    Returns:
        作者索引，未找到返回 -1
    """
    target_parts = normalize_profile_name(name)

    for idx, author in enumerate(authors):
        author_name = author.get("name", "")
        if not author_name:
            continue

        author_parts = normalize_name(author_name)

        if name_match(author_parts, target_parts):
            return idx

    return -1


def get_paper_attr(pub_info: dict, name: str) -> tuple[tuple, int]:
    """从论文信息中提取特征计算所需的属性

    Args:
        pub_info: 论文信息字典，包含 title, venue, keywords, authors 等
        name: 目标作者姓名（下划线分隔格式，如 "wang_wei"）

    Returns:
        ((合作者集合, 机构, 期刊, 关键词, 标题), 作者索引)
    """
    # 提取基本信息
    title = pub_info.get("title", "")
    if title is None:
        title = ""

    venue = pub_info.get("venue", "")
    if venue is None:
        venue = ""

    # 处理关键词
    keywords_data = pub_info.get("keywords", [])
    if keywords_data is None:
        keywords_data = []

    if isinstance(keywords_data, list):
        # 过滤空值和 "null" 字符串
        keywords = " ".join(
            kw for kw in keywords_data if kw and kw != "null" and kw != "None"
        )
    else:
        keywords = str(keywords_data) if keywords_data and keywords_data != "null" else ""

    # 获取作者信息
    authors = pub_info.get("authors", [])
    if authors is None:
        authors = []

    author_idx = get_author_index(name, authors)

    # 提取机构
    org = ""
    if 0 <= author_idx < len(authors):
        org = authors[author_idx].get("org", "")
        if org is None:
            org = ""

    # 收集合作者（排除目标作者）
    coauthors = set()
    for idx, author in enumerate(authors):
        if idx != author_idx:
            author_name = author.get("name", "")
            if author_name:
                coauthors.add(author_name)

    paper_attr = (coauthors, org, venue, keywords, title)

    return paper_attr, author_idx


def get_paper_attr_simple(pub_info: dict) -> tuple:
    """简化版论文属性提取（不需要指定作者姓名）

    提取所有作者作为合作者集合。

    Args:
        pub_info: 论文信息字典

    Returns:
        (合作者集合, 机构, 期刊, 关键词, 标题)
    """
    title = pub_info.get("title", "") or ""
    venue = pub_info.get("venue", "") or ""

    keywords_data = pub_info.get("keywords", []) or []
    if isinstance(keywords_data, list):
        keywords = " ".join(
            kw for kw in keywords_data if kw and kw not in ("null", "None")
        )
    else:
        keywords = str(keywords_data) if keywords_data and keywords_data != "null" else ""

    authors = pub_info.get("authors", []) or []

    coauthors = set()
    orgs = []
    for author in authors:
        name = author.get("name", "")
        if name:
            coauthors.add(name)
        org = author.get("org", "")
        if org:
            orgs.append(org)

    org = orgs[0] if orgs else ""

    return (coauthors, org, venue, keywords, title)


__all__ = ["get_author_index", "get_paper_attr", "get_paper_attr_simple"]
