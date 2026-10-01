"""相似度计算模块

提供统一的相似度计算接口，包括：
- 余弦相似度
- 机构相似度（基于 embedding）
"""

from typing import Callable

import numpy as np

from nallm.core.config import ORG_SIM_THRESHOLD


# Org embedding 获取函数（由调用者注入）
_org_embedding_getter: Callable[[str], np.ndarray | None] | None = None


def set_org_embedding_getter(getter: Callable[[str], np.ndarray | None] | None) -> None:
    """设置 org embedding 获取函数

    Args:
        getter: 接收机构名称，返回 embedding 向量的函数
    """
    global _org_embedding_getter
    _org_embedding_getter = getter


def get_org_embedding(org: str) -> np.ndarray | None:
    """获取机构名称的 embedding 向量

    Args:
        org: 机构名称

    Returns:
        embedding 向量，如果不存在则返回 None
    """
    if not org or not org.strip():
        return None

    if _org_embedding_getter is not None:
        return _org_embedding_getter(org)

    return None


def compute_org_similarity(org1: str, org2: str) -> float:
    """计算两个机构名称的相似度（使用 embedding 向量）

    Args:
        org1: 第一个机构名称
        org2: 第二个机构名称

    Returns:
        相似度值 (0-1)，如果任一机构无效则返回 0.0
    """
    if not org1 or not org2:
        return 0.0

    emb1 = get_org_embedding(org1)
    emb2 = get_org_embedding(org2)

    if emb1 is None or emb2 is None:
        return 0.0

    dot = float(np.dot(emb1, emb2))
    norm1 = float(np.linalg.norm(emb1))
    norm2 = float(np.linalg.norm(emb2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


__all__ = [
    "ORG_SIM_THRESHOLD",
    "set_org_embedding_getter",
    "get_org_embedding",
    "compute_org_similarity",
]
