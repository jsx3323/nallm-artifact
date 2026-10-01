"""姓名匹配模块

提供姓名匹配判断函数。
"""

from nallm.core.config import SINGLE_NAME_PARTS


def is_abbrev(part: str) -> bool:
    """判断姓名部分是否为缩写

    缩写形式：
    - 单字母：'j', 'k'
    - 带点的缩写：'j.', 'k.'

    Args:
        part: 姓名部分

    Returns:
        是否为缩写
    """
    return len(part) == 1 or "." in part


def name_match(name_parts: list[str], profile_parts: list[str]) -> bool:
    """判断姓名是否匹配

    匹配规则：
    - 缩写部分（单字母或带点）：匹配首字母
    - 非缩写部分：精确匹配（避免子串误匹配）

    通过条件：
    - 至少 2 个部分匹配
    - 或单部分姓名且匹配
    - 或特殊单名处理（yamaguchi, martini, hirano）

    Args:
        name_parts: 待消歧论文作者姓名部分列表
        profile_parts: 作者档案姓名部分列表

    Returns:
        是否匹配
    """
    matches = 0
    for part in name_parts:
        if is_abbrev(part):
            # 缩写：匹配首字母
            if any(pp.startswith(part[0]) for pp in profile_parts):
                matches += 1
        else:
            # 非缩写：精确匹配
            if any(part == pp for pp in profile_parts):
                matches += 1

    return (
        matches >= 2
        or (matches == 1 and len(name_parts) == 1)
        or (matches == 1 and any(p in name_parts for p in SINGLE_NAME_PARTS))
    )


__all__ = ["is_abbrev", "name_match", "SINGLE_NAME_PARTS"]
