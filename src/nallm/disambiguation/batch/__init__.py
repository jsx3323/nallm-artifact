"""消歧批处理引擎(纯逻辑)。

下沉自 scripts/run_llm_first.py。本包提供单条/批量消歧的纯编排逻辑,
通过 on_save / on_progress 注入 seam 与上层解耦。
上层编排(端到端入口 API、文件 IO、渲染)由 nallm.pipelines 提供。
"""

from nallm.disambiguation.batch.context import DisambiguationContext
from nallm.disambiguation.batch.engine import (
    batch_disambiguate,
    batch_estimate_tokens,
    disambiguate_record,
    estimate_record_tokens,
)
from nallm.disambiguation.batch.results import (
    ERROR_SENTINEL,
    NONE_AUTHOR_ID,
    _format_time,
    _handle_exception,
    _make_error_result,
    auto_assign_strong,
    build_ground_truth_index,
    get_ground_truth_author_id,
    make_result_dict,
)
from nallm.disambiguation.batch.types import ProgressSnapshot

__all__ = [
    "DisambiguationContext",
    "ProgressSnapshot",
    "auto_assign_strong",
    "batch_disambiguate",
    "batch_estimate_tokens",
    "build_ground_truth_index",
    "disambiguate_record",
    "estimate_record_tokens",
    "get_ground_truth_author_id",
    "make_result_dict",
    "ERROR_SENTINEL",
    "NONE_AUTHOR_ID",
    "_format_time",
    "_handle_exception",
    "_make_error_result",
]
