"""消歧流水线编排层。

提供两条链路的端到端编排 API:
- llm_first: run_llm_first / run_llm_first_from_path
- rule_first: run_rule_first + step_classify/step_llm_co/step_llm_noco/step_merge

以及可单步调用的 run_llm_co / run_llm_noco / merge_predictions。
scripts 层退化为薄 CLI。
"""

from nallm.pipelines.llm_co import run_llm_co
from nallm.pipelines.llm_first import (
    LlmFirstConfig,
    LlmFirstRun,
    RecordFilter,
    run_llm_first,
    run_llm_first_from_path,
)
from nallm.pipelines.merge import merge_predictions, top1_aid
from nallm.pipelines.noco import run_llm_noco
from nallm.pipelines.progress import make_append_store, render_ansi_status
from nallm.pipelines.rule_first import (
    RuleFirstPaths,
    run_rule_first,
    step_classify,
    step_llm_co,
    step_llm_noco,
    step_merge,
)

__all__ = [
    # llm_first
    "LlmFirstConfig",
    "LlmFirstRun",
    "RecordFilter",
    "run_llm_first",
    "run_llm_first_from_path",
    # rule_first
    "RuleFirstPaths",
    "run_rule_first",
    "step_classify",
    "step_llm_co",
    "step_llm_noco",
    "step_merge",
    # 单步
    "run_llm_co",
    "run_llm_noco",
    "merge_predictions",
    "top1_aid",
    # 回调工具
    "make_append_store",
    "render_ansi_status",
]
