"""特征计算核心函数

提供相似度聚合和手工特征计算功能，供 main_feat.py 和 postprocess 模块共同引用。

特征维度:
- 相似度聚合特征 (7维): title_sim_max, venue_sim_max/avg, coauthor_count_max,
  coauthor_ratio_max, org_sim_max, coauthor_org_match_max
- 手工特征 (11维): 从 36 维中筛选的核心特征

总计 18 维特征。
"""

import numpy as np

from nallm.hand_feat import HandFeatureExtractor, get_paper_attr

# 核心手工特征索引 (从 36 维中筛选 11 维)
# 移除: paper_coauthor_tfidf_ratio (1) - 与 coauthor_tfidf 冗余
# 移除: venue_score/ratio (16-19) - 与 venue_sim 冗余
# 移除: 关键词 (30-35) - 三步验证均为冗余
CORE_FEATURE_INDICES = [
    # 合作者 (0-3): 移除 paper_coauthor_tfidf_ratio (1)
    0, 2, 3,  # coauthor_tfidf, counted_coauthor_tfidf, author_coauthor_tfidf_ratio
    # 机构 (4-11): 保留 ratio 特征
    9, 11,  # org_ratio_paper, org_ratio_author
    # 标题 (20-27): 保留 jaccard + score/ratio
    22, 23, 24, 25, 26, 27,  # title_max_jaccard, title_mean_jaccard, title_score_paper, title_ratio_paper, title_score_author, title_ratio_author
]

# 相似度指标配置
# 格式: (指标名称, 是否保留max, 是否保留avg)
# 移除 abstract_sim: 与 title_sim 信息重叠，三步验证均为冗余
SIM_METRIC_CONFIG = [
    ("title_sim", True, False),       # max=0.985, avg 与 max 冗余 (r=0.796)，精简验证后移除 avg
    ("venue_sim", True, True),        # max=0.897, avg=0.786, 差异大，都保留
    ("coauthor_count", True, False),  # max=0.969, avg=0.967, 差异小，只保留 max
    ("coauthor_ratio", True, False),  # max=0.969, avg=0.967, 差异小，只保留 max
    ("org_sim", True, False),         # max=0.769, avg=0.662, avg 弱特征，只保留 max
    ("coauthor_org_match_count", True, False),  # max=0.844, avg=0.844, 差异小，只保留 max
]

# 从配置推导的特征维度数
SIM_FEATURE_DIM = sum(
    (1 if keep_max else 0) + (1 if keep_avg else 0)
    for _, keep_max, keep_avg in SIM_METRIC_CONFIG
)
HAND_FEATURE_DIM = len(CORE_FEATURE_INDICES)


def aggregate_sim_features(pub_sims: dict[str, dict]) -> list[float]:
    """聚合论文级别相似度到候选作者级别

    根据配置选择性计算 max 和 avg。

    Args:
        pub_sims: {pub_id: {title_sim, abstract_sim, ...}}

    Returns:
        SIM_FEATURE_DIM 维聚合特征
    """
    if not pub_sims:
        return [0.0] * SIM_FEATURE_DIM

    features = []

    for metric, keep_max, keep_avg in SIM_METRIC_CONFIG:
        values = []
        for _, sim_data in pub_sims.items():
            val = sim_data.get(metric)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)

        if values:
            max_v = float(max(values))
            avg_v = float(sum(values) / len(values))
        else:
            max_v, avg_v = 0.0, 0.0

        if keep_max:
            features.append(max_v)
        if keep_avg:
            features.append(avg_v)

    return features


def compute_hand_features(
    unass_paper: dict,
    candidate_pubs: list[dict],
    candidate_name: str,
    extractor: HandFeatureExtractor,
) -> list[float]:
    """计算手工特征

    Args:
        unass_paper: 待消歧论文信息
        candidate_pubs: 候选作者论文列表
        candidate_name: 候选作者姓名
        extractor: 特征提取器

    Returns:
        HAND_FEATURE_DIM 维核心特征向量
    """
    if not candidate_name:
        return [0.0] * HAND_FEATURE_DIM

    # 标准化姓名格式
    name_normalized = candidate_name.replace(" ", "_")

    # 获取待消歧论文属性
    try:
        unass_attr, _ = get_paper_attr(unass_paper, name_normalized)
    except Exception:
        return [0.0] * HAND_FEATURE_DIM

    # 获取候选作者论文属性
    candidate_attrs = []
    for pub in candidate_pubs[:50]:  # 限制论文数量
        try:
            attr, _ = get_paper_attr(pub, name_normalized)
            candidate_attrs.append(attr)
        except Exception:
            continue

    if not candidate_attrs:
        return [0.0] * HAND_FEATURE_DIM

    # 计算 36 维特征
    try:
        all_features, _ = extractor.process_ranking_feature(
            (unass_attr, candidate_attrs)
        )
    except Exception:
        return [0.0] * HAND_FEATURE_DIM

    # 筛选核心特征
    core_features = [float(all_features[i]) for i in CORE_FEATURE_INDICES]

    return core_features


def extract_features_from_record(record: dict) -> list:
    """从 JSONL 记录中提取特征向量

    支持两种格式：
    1. 完整特征：sim_features + hand_features
    2. 简化特征：features 字段

    Args:
        record: JSONL 单行解析后的字典

    Returns:
        特征向量列表
    """
    if "features" in record:
        return record["features"]
    feats = record.get("sim_features", []) + record.get("hand_features", [])
    # 含 org_available 字段时追加第 19 维
    if "org_available" in record:
        feats.append(float(record.get("org_available", 0.0)))
    return feats


def load_features(jsonl_paths: list) -> tuple[np.ndarray, np.ndarray]:
    """加载 JSONL 格式的特征数据（支持多文件合并）

    Args:
        jsonl_paths: 特征文件路径列表（Path 对象）

    Returns:
        (features, labels) 数组
    """
    import json

    all_features = []
    all_labels = []

    for jsonl_path in jsonl_paths:
        features = []
        labels = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                features.append(extract_features_from_record(record))
                labels.append(record.get("label", 0))
        all_features.append(np.array(features, dtype=np.float64))
        all_labels.append(np.array(labels, dtype=np.int32))

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


__all__ = [
    "SIM_METRIC_CONFIG",
    "CORE_FEATURE_INDICES",
    "SIM_FEATURE_DIM",
    "HAND_FEATURE_DIM",
    "aggregate_sim_features",
    "compute_hand_features",
    "extract_features_from_record",
    "load_features",
]
