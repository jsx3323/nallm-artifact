"""姓名标准化模块

提供姓名标准化函数。
"""

from nallm.core.config import SINGLE_NAME_PARTS


def remove_ascii(s: str) -> str:
    """移除字符串中的 ASCII 字符"""
    return "".join(ch for ch in s if not ch.isascii())


def normalize_name(name: str) -> list[str]:
    """标准化姓名为列表形式

    Args:
        name: 待标准化的姓名

    Returns:
        标准化后的姓名部分列表
    """
    name = name.strip().replace("\xa0", " ").replace("‐", "").replace("ü", "u")

    if not name.isascii():
        name = remove_ascii(name)
        return [s.lower().strip() for s in name]

    if "GS" in name:
        name = name.replace("GS", "G S")

    name_s = (
        name.lower()
        .replace("-", "")
        .replace("\xa0", " ")
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
        .split(" ")
    )
    return [s.lower() for s in name_s]


def normalize_name_key(name: str) -> str:
    """标准化姓名为字符串键（用于比较）"""
    return "_".join(normalize_name(name))


def normalize_profile_name(name: str) -> list[str]:
    """标准化作者档案姓名（下划线分隔格式）

    WhoIsWho 数据集中的作者档案姓名格式为 "lastname_firstname"，
    此函数将其拆分为列表。

    Args:
        name: 下划线分隔的姓名字符串

    Returns:
        姓名部分列表
    """
    return name.lower().replace("-", "").split("_")


__all__ = [
    "remove_ascii",
    "normalize_name",
    "normalize_name_key",
    "normalize_profile_name",
    "SINGLE_NAME_PARTS",
]
