"""全局配置和常量定义

集中管理项目中使用的配置和常量，避免重复定义和魔法数字。

用法:
    from nallm.core.config import DATA_DIR, ORG_SIM_THRESHOLD
"""

import os
from pathlib import Path

# ============== 项目信息 ==============
__version__ = "2.0.0"

# ============== 相似度阈值 ==============
ORG_SIM_THRESHOLD = 0.7  # 机构相似度阈值

# ============== 子类型分类阈值 ==============
# P1 强度分类阈值
P1_STRONG_TITLE_SIM = 0.9  # Strong 需要的标题相似度
P1_STRONG_COAUTHOR = 3  # Strong 需要的合作者匹配数
P1_HIGH_TITLE_SIM = 0.85  # High 需要的标题相似度
P1_HIGH_COAUTHOR = 2  # High 需要的合作者匹配数
P1_MEDIUM_TITLE_SIM = 0.75  # Medium 需要的标题相似度

# P0 相似度分类阈值
P0_HIGH_TITLE_SIM = 0.85  # P0-High 标题相似度阈值
P0_MEDIUM_TITLE_SIM = 0.75  # P0-Medium 标题相似度阈值

# P0 双重阈值（title_sim_max + top5_avg）
P0_HIGH_TOP5_AVG = 0.80  # P0-High 需要的 top5_avg 阈值
P0_MEDIUM_TOP5_AVG = 0.70  # P0-Medium 需要的 top5_avg 阈值

# P0-High 三重条件阈值（title_sim + org_sim + venue_sim）
P0_HIGH_ORG_SIM = 0.80  # P0-High 需要的 org_sim 阈值
P0_HIGH_VENUE_SIM = 0.85  # P0-High 需要的 venue_sim 阈值

# 竞争场景阈值
COMP_ORG_SIM_THRESHOLD = 0.80  # 竞争场景机构相似度阈值
COMP_VENUE_SIM_THRESHOLD = 0.85  # 竞争场景期刊相似度阈值

# ============== 论文排序阈值 ==============
PUB_ORG_SIM_THRESHOLD = 0.7  # 论文排序机构相似度阈值
PUB_VENUE_SIM_THRESHOLD = 0.75  # 论文排序期刊相似度阈值

# ============== 长度限制 ==============
ORG_MAX_LENGTH = 150  # 机构名称最大长度
MAX_TEXT_LENGTH = 1024  # 文本最大长度（用于 embedding）

# ============== Embedding 配置 ==============
EMBEDDING_DIM = 2560  # Embedding 向量维度
EMBEDDING_BATCH_SIZE = 32  # 批量 embedding 默认批次大小

# Embedding 模型配置（支持环境变量覆盖）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-4B")
EMBEDDING_API_BASE = os.getenv("LOCAL_QWEN3_EMBEDDING_API_BASE", "")

# ============== 路径配置 ==============
# 数据目录
DATA_DIR = Path("data/v3-rnd")

# Embedding 数据库路径
EMBEDDING_DB_DIR = Path("database/embedding")

# 结果输出目录
RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs")

# ============== 姓名匹配特殊集合 ==============
SINGLE_NAME_PARTS = {"yamaguchi", "martini", "hirano"}


# ============== 导出列表 ==============
__all__ = [
    # 项目信息
    "__version__",
    # 相似度阈值
    "ORG_SIM_THRESHOLD",
    # P1 分类阈值
    "P1_STRONG_TITLE_SIM",
    "P1_STRONG_COAUTHOR",
    "P1_HIGH_TITLE_SIM",
    "P1_HIGH_COAUTHOR",
    "P1_MEDIUM_TITLE_SIM",
    # P0 分类阈值
    "P0_HIGH_TITLE_SIM",
    "P0_MEDIUM_TITLE_SIM",
    "P0_HIGH_TOP5_AVG",
    "P0_MEDIUM_TOP5_AVG",
    "P0_HIGH_ORG_SIM",
    "P0_HIGH_VENUE_SIM",
    # 竞争场景阈值
    "COMP_ORG_SIM_THRESHOLD",
    "COMP_VENUE_SIM_THRESHOLD",
    # 论文排序阈值
    "PUB_ORG_SIM_THRESHOLD",
    "PUB_VENUE_SIM_THRESHOLD",
    # 长度限制
    "ORG_MAX_LENGTH",
    "MAX_TEXT_LENGTH",
    # Embedding 配置
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_BASE",
    # 路径配置
    "DATA_DIR",
    "EMBEDDING_DB_DIR",
    "RESULTS_DIR",
    "LOGS_DIR",
    # 特殊集合
    "SINGLE_NAME_PARTS",
]
