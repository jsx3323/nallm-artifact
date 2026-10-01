"""消歧上下文:封装共享状态(org_db + sim_data)。

下沉自 scripts/run_llm_first.py。纯数据容器 + 资源工厂。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nallm.data.similarity import load_sim_data
from nallm.embedding import EmbeddingDB


@dataclass
class DisambiguationContext:
    """消歧上下文，封装共享状态

    用于替代全局变量，提高代码可测试性和线程安全性。
    """

    org_db: EmbeddingDB
    sim_data: dict[str, dict[str, dict[str, dict]]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        org_db_path: str,
        sim_csv_path: Path | None = None,
        record_ids: list[str] | None = None,
    ) -> "DisambiguationContext":
        """创建消歧上下文实例

        Args:
            org_db_path: org embedding 数据库路径
            sim_csv_path: 相似度 CSV 文件路径（可选）
            record_ids: 需要加载的记录 ID 列表（可选，用于过滤）

        Returns:
            DisambiguationContext 实例
        """
        org_db = EmbeddingDB(org_db_path, readonly=True)

        sim_data: dict[str, dict[str, dict[str, dict]]] = {}
        if sim_csv_path and sim_csv_path.exists():
            sim_data = load_sim_data(sim_csv_path, record_ids)

        return cls(
            org_db=org_db,
            sim_data=sim_data,
        )

    def close(self) -> None:
        """关闭资源"""
        if self.org_db:
            self.org_db.close()

    def __enter__(self) -> "DisambiguationContext":
        return self

    def __exit__(self, *_) -> None:
        self.close()
