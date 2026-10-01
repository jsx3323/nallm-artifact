"""编排层结果文件 IO(llm_first 结果 schema)。

从 disambiguation.batch.io 迁入。这些函数编码 llm_first 的结果 CSV schema
(unass_record 列)与四元组 record 契约,属 pipeline 级关注点。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from nallm.utils.io import normalize_candidate_fields, safe_write_csv

logger = logging.getLogger(__name__)


def load_processed_records(results_path: Path) -> set[str]:
    """加载已处理的记录ID

    Args:
        results_path: 结果文件路径

    Returns:
        已处理的记录ID集合
    """
    if not results_path.exists():
        return set()

    try:
        df = pd.read_csv(results_path, usecols=["unass_record"])
        return set(df["unass_record"])
    except Exception as e:
        logger.warning(f"加载已有结果失败: {e}")
        return set()


def remove_records_from_results(results_path: Path, records_to_remove: set[str]) -> int:
    """从结果文件中移除指定记录

    Args:
        results_path: 结果文件路径
        records_to_remove: 要移除的记录ID集合

    Returns:
        移除的记录数
    """
    if not results_path.exists() or not records_to_remove:
        return 0

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        logger.warning(f"读取结果文件失败: {e}，将删除整个文件重新处理")
        results_path.unlink()
        return len(records_to_remove)

    original_count = len(df)
    df = df[~df["unass_record"].isin(list(records_to_remove))]
    removed_count = original_count - len(df)

    if removed_count > 0:
        safe_write_csv(df, results_path)
        logger.info(f"从结果文件中移除 {removed_count} 条记录")

    return removed_count


def append_results(results: list[dict], results_path: Path) -> None:
    """追加结果到文件

    Args:
        results: 结果列表
        results_path: 结果文件路径
    """
    if not results:
        return

    df = pd.DataFrame(results)

    if results_path.exists():
        # 追加模式，不写 header
        safe_write_csv(df, results_path, mode="a", header=False)
    else:
        # 新文件，写入 header
        results_path.parent.mkdir(parents=True, exist_ok=True)
        safe_write_csv(df, results_path)


def load_all_records(
    filtered_candidates_path: Path,
) -> list[tuple[str, list[dict], str, dict]]:
    """从 filtered_candidates.json 加载所有记录

    Returns:
        [(unass_record, candidates, subtype, subtype_details), ...]
    """
    with open(filtered_candidates_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for record, info in data.items():
        subtype = info.get("subtype", "P0-Low")  # 使用新的子类型
        candidates = info.get("candidates", [])
        subtype_details = info.get("subtype_details", {})
        # 统一字段名（valid/test 字段不一致）
        if candidates:
            normalize_candidate_fields(candidates)
        records.append((record, candidates, subtype, subtype_details))

    return records
