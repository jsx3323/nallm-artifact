"""嵌入向量数据库模块

使用 LMDB 存储论文嵌入向量，支持高效的键值查询。
使用 NumPy 二进制格式存储，高效紧凑。

用法:
    from nallm.embedding.database import EmbeddingDB

    with EmbeddingDB("results/embeddings.lmdb") as db:
        db.save("paper_id", [0.1, 0.2, ...])
        vec = db.get("paper_id")  # 返回 np.ndarray
"""

import logging
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingDB:
    """LMDB 嵌入向量数据库封装

    提供嵌入向量的存储和查询功能。
    使用 NumPy float32 二进制格式存储，每个向量占用 10KB。
    """

    def __init__(
        self, db_path: str, map_size: int = 10 * 1024**3, readonly: bool = False
    ):
        """初始化数据库

        Args:
            db_path: 数据库文件路径
            map_size: 数据库最大大小，默认 10GB
            readonly: 是否以只读模式打开，多进程共享时应使用 True
        """
        self.db_path = Path(db_path)
        self.readonly = readonly

        if not readonly:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.env = lmdb.open(
            str(self.db_path),
            map_size=map_size,
            max_dbs=1,
            readonly=readonly,
            lock=not readonly,  # 只读模式不需要锁
        )
        mode = "只读" if readonly else "读写"
        logger.info(f"打开数据库 ({mode}): {self.db_path}")

    def save(self, pid: str, embedding: list[float] | None) -> None:
        """保存单个嵌入向量

        Args:
            pid: 论文 ID
            embedding: 嵌入向量，如果为 None 则删除该键
        """
        with self.env.begin(write=True) as txn:
            if embedding is None:
                txn.delete(pid.encode())
            else:
                arr = np.array(embedding, dtype=np.float32)
                txn.put(pid.encode(), arr.tobytes())

    def save_batch(self, embeddings: dict[str, list[float] | None]) -> None:
        """批量保存嵌入向量

        Args:
            embeddings: 论文 ID 到嵌入向量的映射
        """
        with self.env.begin(write=True) as txn:
            for pid, embedding in embeddings.items():
                if embedding is None:
                    txn.delete(pid.encode())
                else:
                    arr = np.array(embedding, dtype=np.float32)
                    txn.put(pid.encode(), arr.tobytes())
        logger.info(f"批量保存 {len(embeddings)} 个嵌入向量")

    def get(self, pid: str) -> np.ndarray | None:
        """获取单个嵌入向量

        Args:
            pid: 论文 ID

        Returns:
            嵌入向量 (np.ndarray, float32)，如果不存在则返回 None
        """
        with self.env.begin() as txn:
            data = txn.get(pid.encode())
            if data is None:
                return None
            return np.frombuffer(data, dtype=np.float32).copy()

    def get_batch(self, pids: list[str]) -> dict[str, np.ndarray | None]:
        """批量获取嵌入向量

        Args:
            pids: 论文 ID 列表

        Returns:
            论文 ID 到嵌入向量的映射
        """
        result = {}
        with self.env.begin() as txn:
            for pid in pids:
                data = txn.get(pid.encode())
                result[pid] = (
                    np.frombuffer(data, dtype=np.float32).copy() if data else None
                )
        return result

    def __contains__(self, pid: str) -> bool:
        """检查论文 ID 是否存在"""
        with self.env.begin() as txn:
            return txn.get(pid.encode()) is not None

    def __len__(self) -> int:
        """返回数据库中的条目数"""
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def close(self) -> None:
        """关闭数据库"""
        if self.env:
            self.env.close()
            logger.info(f"关闭数据库: {self.db_path}")

    def __enter__(self) -> "EmbeddingDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class CachedEmbeddingDB:
    """带 LRU 缓存的嵌入向量数据库包装器

    用于需要频繁查询同一 key 的场景，减少 LMDB 查询开销。
    适用于 profile embedding 等低频但需要缓存的数据。
    """

    def __init__(self, db_path: str, cache_size: int = 100000):
        """初始化缓存包装器

        Args:
            db_path: 数据库文件路径
            cache_size: LRU 缓存大小，默认 100000 条
        """
        self.db_path = db_path
        self._db: EmbeddingDB | None = None
        self._cache_size = cache_size
        self._get_cached = lru_cache(maxsize=cache_size)(self._get_uncached)

    def _get_uncached(self, pid: str) -> np.ndarray | None:
        """不带缓存的查询方法"""
        if self._db is None:
            self._db = EmbeddingDB(self.db_path, readonly=True)
        return self._db.get(pid)

    def get(self, pid: str) -> np.ndarray | None:
        """获取单个嵌入向量（带缓存）

        Args:
            pid: 论文 ID

        Returns:
            嵌入向量，如果不存在则返回 None
        """
        return self._get_cached(pid)

    def get_batch(self, pids: list[str]) -> dict[str, np.ndarray | None]:
        """批量获取嵌入向量

        Args:
            pids: 论文 ID 列表

        Returns:
            论文 ID 到嵌入向量的映射
        """
        return {pid: self.get(pid) for pid in pids}

    def cache_info(self) -> str:
        """返回缓存统计信息"""
        info = self._get_cached.cache_info()
        hit_rate = (
            info.hits / (info.hits + info.misses)
            if (info.hits + info.misses) > 0
            else 0
        )
        return f"缓存命中: {info.hits}, 未命中: {info.misses}, 命中率: {hit_rate:.2%}, 缓存大小: {info.currsize}"

    def cache_clear(self) -> None:
        """清空缓存"""
        self._get_cached.cache_clear()

    def close(self) -> None:
        """关闭数据库"""
        if self._db:
            self._db.close()
            self._db = None
        self.cache_clear()

    def __enter__(self) -> "CachedEmbeddingDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = ["EmbeddingDB", "CachedEmbeddingDB"]
