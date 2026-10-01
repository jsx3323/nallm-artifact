"""消歧结果构造(纯函数 + 常量)。

下沉自 scripts/run_llm_first.py。
"""

from __future__ import annotations

import asyncio
import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from nallm.disambiguation import (
    DisambiguationOutput,
    EmptyLLMResponseError,
)

logger = logging.getLogger(__name__)

# 结果常量
ERROR_SENTINEL = "error"  # 错误结果的占位符
NONE_AUTHOR_ID = "none"

# 致命错误状态码（不重试，直接终止）
FATAL_STATUS_CODES = (401, 402, 403)


def _format_time(seconds: float) -> str:
    """格式化时间显示

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串，如 "1h 23m 45s" 或 "5m 30s"
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def build_ground_truth_index(data: dict) -> dict[str, str]:
    """构建 pub_id -> author_id 的索引

    将嵌套的 ground_truth 结构扁平化为 O(1) 查找的字典。

    Args:
        data: 原始数据字典

    Returns:
        {pub_id: author_id, ...} 索引字典
    """
    index: dict[str, str] = {}
    valid_ground_truth = data.get("valid_ground_truth", {})

    for authors in valid_ground_truth.values():
        for author_id, pubs in authors.items():
            for pub_id in pubs:
                index[pub_id] = author_id

    return index


def get_ground_truth_author_id(
    unass_record: str, ground_truth_index: dict[str, str]
) -> str:
    """获取真实作者ID（使用预构建索引）"""
    unass_pub_id = unass_record.split("-")[0]
    return ground_truth_index.get(unass_pub_id, NONE_AUTHOR_ID)


def auto_assign_strong(candidates: list[dict]) -> DisambiguationOutput:
    """自动分配 P1-Strong 记录"""
    if not candidates:
        return DisambiguationOutput(
            author_id=NONE_AUTHOR_ID, confidence="high", reasoning="无候选作者"
        )

    top1 = candidates[0]
    aid = top1.get("aid", "none")
    title_sim = top1.get("title_sim", 0)
    coauthor_count = top1.get("coauthor_count", 0)
    org_verified = top1.get("coauthor_org_match_count", 0) > 0

    reasoning = f"自动分配: title_sim={title_sim:.2f}, coauthor={coauthor_count}, 机构验证={org_verified}"

    return DisambiguationOutput(author_id=aid, confidence="high", reasoning=reasoning)


def make_result_dict(
    unass_record: str,
    predicted_author_id: str,
    confidence: str,
    reasoning: str,
    ground_truth: str,
    num_candidates: int,
    subtype: str,
    strategy: str,
    stats: dict,
    estimated_tokens: int = 0,
) -> dict:
    """统一构建结果字典

    Args:
        unass_record: 记录 ID
        predicted_author_id: 预测的作者 ID
        confidence: 置信度
        reasoning: 推理说明
        ground_truth: 真实作者 ID
        num_candidates: 候选数量
        subtype: 子类型
        strategy: 策略
        stats: 统计信息字典
        estimated_tokens: 预估 token 数

    Returns:
        结果字典
    """
    return {
        "unass_record": unass_record,
        "predicted_author_id": predicted_author_id,
        "confidence": confidence,
        "reasoning": reasoning,
        "ground_truth_author_id": ground_truth,
        "num_candidates": num_candidates,
        "subtype": subtype,
        "strategy": strategy,
        "top1_title_sim": stats.get("max_title_sim", 0),
        "top1_coauthor_count": stats.get("total_coauthor_count", 0),
        "top1_org_sim": stats.get("max_org_sim", 0),
        "top1_venue_sim": stats.get("max_venue_sim", 0),
        "estimated_prompt_tokens": estimated_tokens,
    }


def _make_error_result(
    record: str,
    candidates: list[dict],
    subtype: str,
    ground_truth_index: dict[str, str],
    error_type: str,
    error_msg: str,
) -> dict:
    """构建错误结果字典

    Args:
        record: 记录 ID
        candidates: 候选作者列表
        subtype: 子类型
        ground_truth_index: pub_id -> author_id 索引
        error_type: 错误类型 (rate_limit/timeout/connection/api_error/unknown)
        error_msg: 错误信息

    Returns:
        错误结果字典
    """
    return {
        "unass_record": record,
        "predicted_author_id": ERROR_SENTINEL,
        "confidence": ERROR_SENTINEL,
        "reasoning": f"[{error_type}] {error_msg}",
        "ground_truth_author_id": get_ground_truth_author_id(
            record, ground_truth_index
        ),
        "num_candidates": len(candidates),
        "subtype": subtype,
        "strategy": ERROR_SENTINEL,
        "top1_title_sim": 0,
        "top1_coauthor_count": 0,
        "top1_org_sim": 0,
        "top1_venue_sim": 0,
        "estimated_prompt_tokens": 0,
    }


def _handle_exception(
    e: Exception,
    record: str,
    candidates: list[dict],
    subtype: str,
    ground_truth_index: dict[str, str],
) -> dict | None:
    """处理异常，返回错误结果。致命错误返回 None 让调用方重新抛出。

    Args:
        e: 捕获的异常
        record: 记录 ID
        candidates: 候选作者列表
        subtype: 子类型
        ground_truth_index: pub_id -> author_id 索引

    Returns:
        错误结果字典，致命错误返回 None
    """
    # 超时错误特殊处理
    if isinstance(e, (asyncio.TimeoutError, APITimeoutError)):
        error_type, error_desc = "timeout", "请求超时"
    elif isinstance(e, RateLimitError):
        error_type, error_desc = "rate_limit", "API 限流"
    elif isinstance(e, APIConnectionError):
        error_type, error_desc = "connection", "连接错误"
    elif isinstance(e, EmptyLLMResponseError):
        error_type, error_desc = "empty_response", "LLM 空响应"
    elif isinstance(e, APIStatusError):
        if e.status_code in FATAL_STATUS_CODES:
            logger.critical(f"致命错误 ({e.status_code}): {str(e)}")
            return None
        error_type, error_desc = "api_error", f"API 错误 ({e.status_code})"
    elif isinstance(e, ValueError):
        error_type, error_desc = "data_error", "数据错误"
    elif isinstance(e, RuntimeError):
        error_type, error_desc = "runtime_error", "运行时错误"
    else:
        error_type = "unknown"
        error_desc = f"未知错误 ({type(e).__name__})"

    error_msg = f"[{record}] {error_desc}: {str(e)[:100]}"
    logger.error(error_msg)
    return _make_error_result(
        record, candidates, subtype, ground_truth_index, error_type, error_msg
    )
