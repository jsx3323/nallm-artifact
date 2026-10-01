"""后处理场景的特征查询层

从预计算的 JSONL 特征文件中查表获取特征，避免实时计算。
"""

import json
import logging
from pathlib import Path

import numpy as np

from nallm.feature.compute import SIM_FEATURE_DIM, HAND_FEATURE_DIM, extract_features_from_record

logger = logging.getLogger(__name__)

FEATURE_DIM = SIM_FEATURE_DIM + HAND_FEATURE_DIM


class FeatureLookup:
    """从 JSONL 特征文件查表获取候选特征

    特征文件格式: 每行一个 JSON 对象
    {
        "unass_record": "...",
        "candidate_aid": "...",
        "sim_features": [...],
        "hand_features": [...],
        "label": 0/1,
        "pub_count": N
    }
    """

    def __init__(self):
        # {unass_record: {candidate_aid: features_ndarray}}
        self._data: dict[str, dict[str, np.ndarray]] = {}

    def load(self, jsonl_path: Path) -> None:
        """加载 JSONL 特征文件

        Args:
            jsonl_path: 特征文件路径
        """
        logger.info(f"加载特征数据: {jsonl_path}")
        count = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                unass_record = record["unass_record"]
                aid = record["candidate_aid"]
                features = extract_features_from_record(record)

                if unass_record not in self._data:
                    self._data[unass_record] = {}
                self._data[unass_record][aid] = np.array(features, dtype=np.float64)
                count += 1

        logger.info(f"  加载完成: {len(self._data)} 条记录, {count} 个候选")

    def get_candidates_features(
        self, unass_record: str
    ) -> list[tuple[str, np.ndarray]]:
        """获取一条记录所有候选的特征

        Args:
            unass_record: 待消歧记录 ID

        Returns:
            [(candidate_aid, features), ...] 列表
        """
        record_data = self._data.get(unass_record, {})
        return list(record_data.items())

    def candidate_count(self, unass_record: str) -> int:
        """返回一条记录在特征文件中的候选数

        Args:
            unass_record: 待消歧记录 ID

        Returns:
            候选数，特征文件中无该记录时返回 0
        """
        record_data = self._data.get(unass_record, {})
        return len(record_data)

    def get_candidate_feature(
        self, unass_record: str, candidate_aid: str
    ) -> np.ndarray | None:
        """获取单个候选的特征

        Args:
            unass_record: 待消歧记录 ID
            candidate_aid: 候选作者 ID

        Returns:
            (FEATURE_DIM,) 特征向量，不存在返回 None
        """
        record_data = self._data.get(unass_record, {})
        return record_data.get(candidate_aid)


__all__ = [
    "FEATURE_DIM",
    "FeatureLookup",
]
