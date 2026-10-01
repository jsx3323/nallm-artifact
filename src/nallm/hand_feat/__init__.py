"""手工特征提取模块

提供 36 维手工特征的计算功能。
"""

from nallm.hand_feat.config import (
    IDF_DIR,
    DEFAULT_IDF_VALUES,
    FEATURE_GROUPS,
    FEATURE_NAMES,
)
from nallm.hand_feat.extractor import HandFeatureExtractor
from nallm.hand_feat.name_match import MatchName, clean_name
from nallm.hand_feat.paper_attr import (
    get_author_index,
    get_paper_attr,
    get_paper_attr_simple,
)

__all__ = [
    # Config
    "IDF_DIR",
    "DEFAULT_IDF_VALUES",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    # Extractor
    "HandFeatureExtractor",
    # Name matching
    "MatchName",
    "clean_name",
    # Paper attributes
    "get_author_index",
    "get_paper_attr",
    "get_paper_attr_simple",
]
