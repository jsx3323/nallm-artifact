"""姓名处理模块 - 标准化、匹配、索引生成"""

from nallm.name.normalize import (
    remove_ascii,
    normalize_name,
    normalize_name_key,
    normalize_profile_name,
    SINGLE_NAME_PARTS,
)
from nallm.name.match import is_abbrev, name_match
from nallm.name.index import generate_index_keys

__all__ = [
    "remove_ascii",
    "normalize_name",
    "normalize_name_key",
    "normalize_profile_name",
    "is_abbrev",
    "name_match",
    "generate_index_keys",
    "SINGLE_NAME_PARTS",
]
