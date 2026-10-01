"""论文采样模块

对候选作者的全部论文按 hybrid_score 排序，选 top-k 论文供 LLM 参考。

排序优先级（数据验证 1000 案例 Top-1 命中率 99.4%）：
1. 合作者匹配（co≥2 > co=1 > co=0）
2. 标题/期刊/机构相似度
3. 时间近（年份差越小越好）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from nallm.embedding.compute import cosine_similarity
from nallm.embedding.database import EmbeddingDB

logger = logging.getLogger(__name__)


# Hybrid 评分权重（数据驱动）
CO_BASE_HIGH = 1000  # co≥2
CO_BASE_LOW = 500    # co=1
W_TITLE = 100
W_VENUE = 10
W_ORG = 10
W_YEAR_PENALTY = 5  # 每差 1 年扣分


@dataclass
class PaperSample:
    """采样论文结果"""

    pub: dict
    co: int
    title_sim: float
    venue_sim: float
    org_sim: float
    year_diff: int | None
    score: float


@dataclass
class _NewPubMeta:
    """预计算的待消歧论文规格化数据，避免在循环中重复提取"""
    author_names: set[str]
    venue: str
    org: str
    year: int | None
    paper_id: str = ""


def _precompute_new_pub_meta(
    new_pub: dict, paper_id_override: str | None = None
) -> _NewPubMeta:
    """提前提取待消歧论文的元数据

    Args:
        new_pub: 待消歧论文
        paper_id_override: 显式指定 8 字符 paper id（优先于推断）
    """
    authors = new_pub.get("authors", [])
    anames = {a["name"].lower().strip() for a in authors if a.get("name")}

    # 解析 paper_id：优先 override，否则从 paper_id/id 推断
    if paper_id_override:
        pid = paper_id_override
    else:
        pid = new_pub.get("paper_id", "")
        if not pid:
            nb = new_pub.get("id", "")
            if len(nb) == 8:
                pid = nb

    return _NewPubMeta(
        author_names=anames,
        venue=(new_pub.get("venue") or "").lower(),
        org=(authors[0].get("org") or "").lower() if authors else "",
        year=_year_val(new_pub),
        paper_id=pid,
    )


def _author_names(paper: dict) -> set[str]:
    return {
        a["name"].lower().strip()
        for a in paper.get("authors", [])
        if a.get("name")
    }


def _year_val(paper: dict) -> int | None:
    y = paper.get("year")
    if isinstance(y, int):
        return y
    if isinstance(y, str) and y.isdigit():
        return int(y)
    return None


def year_of(paper: dict) -> int | None:
    """提取论文年份"""
    return _year_val(paper)


def _venue(paper: dict) -> str:
    return (paper.get("venue") or "").lower()


def _org(paper: dict) -> str:
    authors = paper.get("authors", [])
    if not authors:
        return ""
    return (authors[0].get("org") or "").lower()


def jaccard(a: str, b: str) -> float:
    """Jaccard 字符串相似度（基于词集）"""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def co_overlap(paper: dict, new_pub: dict) -> int:
    """共同作者数"""
    return len(_author_names(paper) & _author_names(new_pub))


def venue_sim(paper: dict, new_pub: dict) -> float:
    """期刊 jaccard 相似度"""
    return jaccard(_venue(paper), _venue(new_pub))


def org_sim(paper: dict, new_pub: dict) -> float:
    """机构 jaccard 相似度"""
    return jaccard(_org(paper), _org(new_pub))


def year_diff(paper: dict, new_pub: dict) -> int | None:
    """年份差"""
    yp, yn = _year_val(paper), _year_val(new_pub)
    if yp is None or yn is None:
        return None
    return abs(yp - yn)


def hybrid_score(
    co: int,
    title_sim: float,
    venue_sim: float,
    org_sim: float,
    year_diff: int | None,
) -> float:
    """计算 hybrid_score

    Args:
        co: 共同作者数
        title_sim: 标题相似度 [0, 1]
        venue_sim: 期刊相似度 [0, 1]
        org_sim: 机构相似度 [0, 1]
        year_diff: 年份差

    Returns:
        hybrid_score
    """
    if co >= 2:
        base = CO_BASE_HIGH
    elif co == 1:
        base = CO_BASE_LOW
    else:
        base = 0
    base += W_TITLE * title_sim + W_VENUE * venue_sim + W_ORG * org_sim
    if year_diff is not None:
        base -= W_YEAR_PENALTY * year_diff
    return base


def _title_sim_from_db(pid_p: str, pid_n: str, title_db: EmbeddingDB) -> float:
    """从 embedding DB 查两个 paper ID 的 title 相似度"""
    if not pid_p or not pid_n:
        return 0.0
    v_p = title_db.get(pid_p)
    v_n = title_db.get(pid_n)
    if v_p is None or v_n is None:
        return 0.0
    return cosine_similarity(v_p, v_n)


def _score_paper_with_meta(
    paper: dict,
    meta: _NewPubMeta,
    title_db: EmbeddingDB | None = None,
) -> PaperSample:
    """用预计算的 new_pub meta 计算单篇论文的 hybrid_score（内部函数，避免重复计算）

    Args:
        paper: 候选作者论文
        meta: 预计算的待消歧论文元数据
        title_db: title embedding 数据库（无则 title_sim=0）

    Returns:
        PaperSample 结果
    """
    co = len(_author_names(paper) & meta.author_names)
    vs = jaccard(_venue(paper), meta.venue)
    os_ = jaccard(_org(paper), meta.org)
    yp = _year_val(paper)
    yr = abs(yp - meta.year) if (yp is not None and meta.year is not None) else None

    if title_db:
        pid_p = paper.get("id", "")
        ts = _title_sim_from_db(pid_p, meta.paper_id, title_db) if meta.paper_id else 0.0
    else:
        ts = 0.0

    sc = hybrid_score(co, ts, vs, os_, yr)
    return PaperSample(
        pub=paper,
        co=co,
        title_sim=ts,
        venue_sim=vs,
        org_sim=os_,
        year_diff=yr,
        score=sc,
    )


def sample_papers_for_candidate(
    candidate_pubs: list[dict],
    new_pub: dict,
    top_k: int = 3,
    title_db: EmbeddingDB | None = None,
    new_pub_paper_id: str | None = None,
) -> list[PaperSample]:
    """对候选作者论文按 hybrid_score 排序，返回 top-k

    Args:
        candidate_pubs: 候选作者的全部论文列表
        new_pub: 待消歧论文
        top_k: 返回论文数
        title_db: title embedding 数据库
        new_pub_paper_id: 待消歧论文的 8 字符 paper id（推荐传入）

    Returns:
        按 score 降序的 PaperSample 列表
    """
    if not candidate_pubs:
        return []

    # 提前计算 new_pub 元数据，避免在循环中重复计算
    meta = _precompute_new_pub_meta(new_pub, new_pub_paper_id)
    samples = [_score_paper_with_meta(p, meta, title_db) for p in candidate_pubs]
    samples.sort(key=lambda s: -s.score)
    return samples[:top_k]


__all__ = [
    "PaperSample",
    "jaccard",
    "co_overlap",
    "venue_sim",
    "org_sim",
    "year_diff",
    "year_of",
    "hybrid_score",
    "sample_papers_for_candidate",
]