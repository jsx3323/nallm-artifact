"""向量嵌入模块 - 嵌入计算、数据库封装"""

from nallm.embedding.compute import (
    get_text_embedding,
    get_text_embeddings_batch,
    cosine_similarity,
)
from nallm.embedding.database import EmbeddingDB, CachedEmbeddingDB

__all__ = [
    "get_text_embedding",
    "get_text_embeddings_batch",
    "cosine_similarity",
    "EmbeddingDB",
    "CachedEmbeddingDB",
]
