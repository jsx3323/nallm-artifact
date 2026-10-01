"""数据加载工具模块

提供 JSON/CSV/GT 等常见数据格式的统一加载函数。

用法:
    from nallm.utils.io import load_json, load_csv_predictions, load_gt
"""

import csv
import json
from pathlib import Path
from typing import Any

from nallm.core.config import DATA_DIR


def load_json(path: str | Path) -> Any:
    """加载 JSON 文件

    Args:
        path: JSON 文件路径

    Returns:
        解析后的 Python 对象（dict/list）
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_predictions(
    csv_path: str | Path,
    uid_col: str = "uid",
    pred_col: str = "predicted_aid",
) -> dict[str, str]:
    """从 CSV 加载预测结果

    Args:
        csv_path: CSV 文件路径
        uid_col: UID 列名
        pred_col: 预测值列名

    Returns:
        {uid: pred} 字典
    """
    preds = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds[row[uid_col]] = row[pred_col]
    return preds


def load_gt(split: str = "valid") -> dict[str, str]:
    """加载指定 split 的 GT（ground truth）

    Args:
        split: "valid" 或 "test"

    Returns:
        {uid: gt_author_id} 字典
    """
    gt_path = DATA_DIR / split / f"cna_{split}_unass_gt.json"
    return load_json(gt_path) if gt_path.exists() else {}


def load_classified(split: str = "valid") -> tuple[dict[str, str], dict[str, list[str]]]:
    """加载分类结果

    Args:
        split: "valid" 或 "test"

    Returns:
        (uid_to_subtype, subtype_to_uids)
        - uid_to_subtype: {uid: subtype}
        - subtype_to_uids: {subtype: [uid, ...]}
    """
    classified_path = Path("results") / f"{split}_unass_classified.csv"
    uid_to_subtype = {}
    subtype_to_uids = {}

    if not classified_path.exists():
        return uid_to_subtype, subtype_to_uids

    with open(classified_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["unass_record"]
            sub = row["subtype"]
            uid_to_subtype[uid] = sub
            subtype_to_uids.setdefault(sub, []).append(uid)

    return uid_to_subtype, subtype_to_uids


def load_filtered_candidates(split: str = "valid") -> dict:
    """加载过滤后的候选数据"""
    path = Path("results") / f"{split}_filtered_candidates.json"
    return load_json(path) if path.exists() else {}


# ============== CSV 安全写入 ==============

def safe_write_csv(df, path: str | Path, **kwargs) -> None:
    """安全写入 CSV，自动处理推理字段中的逗号

    所有消歧结果 CSV 写入应通过此函数，确保引用一致性。

    Args:
        df: pandas DataFrame
        path: 输出路径
        **kwargs: 传递给 df.to_csv 的额外参数（可覆盖默认 quoting）
    """
    kwargs.setdefault("index", False)
    kwargs.setdefault("quoting", csv.QUOTE_NONNUMERIC)
    df.to_csv(path, **kwargs)


# ============== 候选字段标准化 ==============

def normalize_candidate_fields(candidates: list[dict]) -> None:
    """原地标准化候选字段名（处理 valid/test 数据集字段不一致）

    Args:
        candidates: 候选列表，每项可有 candidate_aid/aid/id 等变体
    """
    for cand in candidates:
        # 统一 aid 字段：test 用 candidate_aid，valid 用 aid
        if "candidate_aid" in cand and "aid" not in cand:
            cand["aid"] = cand["candidate_aid"]


__all__ = [
    "load_json",
    "load_csv_predictions",
    "load_gt",
    "load_classified",
    "load_filtered_candidates",
    # 安全写入
    "safe_write_csv",
    # 字段标准化
    "normalize_candidate_fields",
]
