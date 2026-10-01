"""分类模块 - 子类型分类、置信度计算、候选排序"""

from nallm.classification.subtype_llm_first import (
    classify_llm_first,
    classify_l1_detailed,
    get_matched_coauthors_with_org,
)
from nallm.classification.confidence import (
    compute_confidence,
    calibrate_confidence,
    CONFIDENCE_ORG_SIM_HIGH,
    CONFIDENCE_ORG_SIM_STRONG,
    CONFIDENCE_ORG_SIM_P0_MEDIUM,
    CONFIDENCE_VENUE_SIM_HIGH,
    CONFIDENCE_VENUE_SIM_EXEMPTION,
    CONFIDENCE_ORG_SIM_NO_MATCH,
    CONFIDENCE_TITLE_SIM_STRONG,
    CONFIDENCE_COAUTHOR_MIN,
    CONFIDENCE_COAUTHOR_HIGH,
    CONFIDENCE_COAUTHOR_EXCEPTION,
    CONFIDENCE_ORDER,
)
from nallm.classification.candidates import (
    candidate_sort_key_v3,
    sort_candidates_v3,
    format_candidates,
)

__all__ = [
    # Subtype
    "classify_llm_first",
    "classify_l1_detailed",
    "get_matched_coauthors_with_org",
    # Confidence
    "compute_confidence",
    "calibrate_confidence",
    "CONFIDENCE_ORG_SIM_HIGH",
    "CONFIDENCE_ORG_SIM_STRONG",
    "CONFIDENCE_ORG_SIM_P0_MEDIUM",
    "CONFIDENCE_VENUE_SIM_HIGH",
    "CONFIDENCE_VENUE_SIM_EXEMPTION",
    "CONFIDENCE_ORG_SIM_NO_MATCH",
    "CONFIDENCE_TITLE_SIM_STRONG",
    "CONFIDENCE_COAUTHOR_MIN",
    "CONFIDENCE_COAUTHOR_HIGH",
    "CONFIDENCE_COAUTHOR_EXCEPTION",
    "CONFIDENCE_ORDER",
    # Candidates
    "candidate_sort_key_v3",
    "sort_candidates_v3",
    "format_candidates",
]

