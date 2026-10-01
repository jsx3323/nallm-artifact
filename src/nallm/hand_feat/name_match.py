"""姓名匹配适配层

适配 hand_feat.py 的 MatchName 接口，内部使用 nallm.name.name_match。

原接口: MatchName(paper_names, author_names, name2clean, return_set)
现接口: name_match(name_parts, profile_parts)
"""

from nallm.name import normalize_name, name_match


def clean_name(name: str) -> str:
    """清洗姓名，返回标准化格式

    Args:
        name: 原始姓名

    Returns:
        清洗后的姓名（空格分隔的小写形式）
    """
    return " ".join(normalize_name(name))


def MatchName(
    paper_names: list[str],
    author_names: list[str],
    name2clean: dict[str, str],
    return_set: bool = True,
) -> set[str]:
    """匹配论文合作者与作者历史论文的合作者

    适配 hand_feat.py 的调用接口。

    Args:
        paper_names: 待消歧论文的合作者姓名列表
        author_names: 候选作者历史论文的合作者姓名列表
        name2clean: 姓名清洗缓存字典（会被更新）
        return_set: 是否返回集合（保持接口兼容）

    Returns:
        匹配的合作者清洗后姓名集合
    """
    matched_coauthors = set()

    # 清洗并缓存姓名
    for name in paper_names:
        if name not in name2clean:
            name2clean[name] = clean_name(name)

    for name in author_names:
        if name not in name2clean:
            name2clean[name] = clean_name(name)

    # 遍历匹配
    for paper_name in paper_names:
        paper_parts = normalize_name(paper_name)

        for author_name in author_names:
            author_parts = normalize_name(author_name)

            if name_match(paper_parts, author_parts):
                # 使用清洗后的姓名作为匹配结果
                matched_coauthors.add(name2clean[paper_name])
                break  # 一个 paper_name 只需匹配一次

    return matched_coauthors if return_set else matched_coauthors


__all__ = ["MatchName", "clean_name"]
