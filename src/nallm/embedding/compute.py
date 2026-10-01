"""文本嵌入计算模块

提供文本嵌入向量的计算功能。

用法:
    from nallm.embedding.compute import get_text_embedding, cosine_similarity
"""

import logging
import os
from typing import Union

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial.distance import cosine
from tqdm import tqdm

from nallm.core.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_API_BASE,
    MAX_TEXT_LENGTH,
)

# 配置日志
logger = logging.getLogger(__name__)

# 从 .env 文件加载环境变量
load_dotenv()

# 初始化 OpenAI 客户端，用于调用语言模型进行文本嵌入
# api_key 走环境变量,新版 openai 对显式空 key import 即炸;
# 此 client 仅服务 API 调用,本地计算路径零改动
# 惰性单例:import 阶段零凭证也不炸,首次调用 API 时才构造
_embedding_client: OpenAI | None = None


def _get_embedding_client() -> OpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=EMBEDDING_API_BASE or os.getenv("LOCAL_QWEN3_EMBEDDING_API_BASE"),
        )
    return _embedding_client


def get_text_embedding(text: str) -> list[float] | None:
    """获取文本的嵌入向量

    Args:
        text: 待计算嵌入向量的文本

    Returns:
        嵌入向量，如果输入无效则返回 None
    """
    if not isinstance(text, str) or len(text) == 0:
        return None

    # 截断文本
    text = text[:MAX_TEXT_LENGTH]

    response = _get_embedding_client().embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    embedding = response.data[0].embedding
    return embedding


def get_text_embeddings_batch(
    texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE
) -> list[list[float] | None]:
    """批量获取文本的嵌入向量

    Args:
        texts: 待计算嵌入向量的文本列表
        batch_size: 每批处理的文本数量，默认32

    Returns:
        嵌入向量列表，与输入文本顺序对应。如果某个文本无效，对应位置返回 None
    """
    if not texts:
        return []

    # 过滤空文本，记录有效文本的索引
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and len(text) > 0:
            valid_indices.append(i)
            # 截断文本
            valid_texts.append(text[:MAX_TEXT_LENGTH])

    # 初始化结果列表，全部为 None
    results: list[list[float] | None] = [None] * len(texts)

    if not valid_texts:
        return results

    # 分批处理
    for i in tqdm(range(0, len(valid_texts), batch_size), desc="批量获取 embedding"):
        batch_texts = valid_texts[i : i + batch_size]
        batch_indices = valid_indices[i : i + batch_size]

        try:
            response = _get_embedding_client().embeddings.create(
                input=batch_texts,
                model=EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIM,
            )
            # 按顺序填充结果
            for j, data in enumerate(response.data):
                results[batch_indices[j]] = data.embedding
        except Exception as e:
            logger.error(f"批量 embedding 计算错误 (batch {i // batch_size}): {e}")
            # 该批次的文本全部标记为 None
            for idx in batch_indices:
                results[idx] = None

    return results


def cosine_similarity(
    vec1: Union[list[float], np.ndarray, None],
    vec2: Union[list[float], np.ndarray, None],
) -> float:
    """计算两个向量之间的余弦相似度

    Args:
        vec1: 第一个向量（支持 list 或 numpy.ndarray）
        vec2: 第二一个向量（支持 list 或 numpy.ndarray）

    Returns:
        余弦相似度，如果任一向量为空则返回 0.0
    """
    if vec1 is None or vec2 is None:
        return 0.0
    return float(1 - cosine(vec1, vec2))


__all__ = [
    "get_text_embedding",
    "get_text_embeddings_batch",
    "cosine_similarity",
]
