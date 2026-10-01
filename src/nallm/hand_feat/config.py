"""手工特征配置

包含路径配置、特征维度定义、特征名称等常量。
"""

from pathlib import Path

# IDF 字典路径
IDF_DIR = Path("data/tf_idf")

# 默认 IDF 值(OOV 词回退值)
# title 降为 4(7.2 根因②):原 14.79 把 OOV 领域词(storage/quenching 等,
# 真实 IDF<6)放大 8-19x,占 score_author 贡献 33.8%;实测 3-6 区间,取中值,
# 评估时可对 3/4/5/6 小网格
DEFAULT_IDF_VALUES = {
    "org": 14.37,
    "venue": 10.42,
    "title": 4.0,
    "keyword": 1.0,
}

# 特征维度定义 (名称, 起始索引, 结束索引)
FEATURE_GROUPS = {
    "coauthor": (0, 4),  # 共同作者特征 (4维)
    "org": (4, 12),  # 机构特征 (8维)
    "venue": (12, 20),  # 期刊特征 (8维)
    "title": (20, 28),  # 标题特征 (8维)
    "keyword": (28, 36),  # 关键词特征 (8维)
}

# 特征名称 (36维)
FEATURE_NAMES = [
    # 共同作者特征 (0-3)
    "coauthor_tfidf",
    "paper_coauthor_tfidf_ratio",
    "counted_coauthor_tfidf",
    "author_coauthor_tfidf_ratio",
    # 机构特征 (4-11)
    "org_max_jaro",
    "org_mean_jaro",
    "org_max_jaccard",
    "org_mean_jaccard",
    "org_score_paper",
    "org_ratio_paper",
    "org_score_author",
    "org_ratio_author",
    # 期刊特征 (12-19)
    "venue_max_jaro",
    "venue_mean_jaro",
    "venue_max_jaccard",
    "venue_mean_jaccard",
    "venue_score_paper",
    "venue_ratio_paper",
    "venue_score_author",
    "venue_ratio_author",
    # 标题特征 (20-27)
    "title_max_jaro",
    "title_mean_jaro",
    "title_max_jaccard",
    "title_mean_jaccard",
    "title_score_paper",
    "title_ratio_paper",
    "title_score_author",
    "title_ratio_author",
    # 关键词特征 (28-35)
    "kw_max_jaro",
    "kw_mean_jaro",
    "kw_max_jaccard",
    "kw_mean_jaccard",
    "kw_score_paper",
    "kw_ratio_paper",
    "kw_score_author",
    "kw_ratio_author",
]

__all__ = [
    "IDF_DIR",
    "DEFAULT_IDF_VALUES",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
]
