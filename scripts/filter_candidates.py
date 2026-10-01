"""
基于 valid_unass_sim_full.csv 筛选候选作者并进行子类型分类

流程：
1. 读取相似度数据
2. 按 unass_record 分组，计算每个候选作者的最大 title_sim
3. 按阈值筛选候选作者
4. 进行统一子类型分类

子类型结构：
- NONE: 无候选作者
- P1-*: Top1 有合作者匹配（Strong/High/Medium/Weak，带 -Comp 后缀）
- P2-*: Top2 有合作者匹配（带 -Comp 后缀）
- P3-*: Top3 有合作者匹配（带 -Comp 后缀）
- P0-*: 无合作者匹配（High/Medium/Low）
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nallm.classification import classify_llm_first
from nallm.utils.stats import compute_topk_avg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.55


def subtype_to_difficulty(subtype: str) -> str:
    """从子类型推导难度等级（向后兼容）

    Args:
        subtype: 子类型字符串

    Returns:
        难度等级 (NONE/L1/L2/L3)
    """
    if subtype == "NONE":
        return "L0"
    elif subtype.startswith("P0"):
        # P0-High/Medium -> L2, P0-Low -> L3
        if subtype == "P0-Low":
            return "L3"
        return "L2"
    else:
        # P1/P2/P3 系列都是 L1
        return "L1"


def classify_record(candidates: list[dict]) -> tuple[str, str, dict]:
    """记录分类

    Args:
        candidates: 筛选后的候选作者列表，按相似度降序排列

    Returns:
        (子类型, 原因说明, 详细信息)
    """
    subtype, details = classify_llm_first(candidates)

    # 构建原因说明
    if subtype == "NONE":
        reason = "无候选作者"
    elif subtype.startswith("P0"):
        title_sim = details.get("title_sim", 0)
        reason = f"无合作者匹配，title_sim={title_sim:.2f}"
    else:
        position = details.get("position", 0)
        title_sim = details.get("title_sim", 0)
        coauthor_count = details.get("coauthor_count", 0)
        has_competition = details.get("has_competition", False)

        if position == 1:
            strength = details.get("strength", "")
            comp_suffix = "-Comp" if has_competition else ""
            reason = f"Top1 有合作者匹配({coauthor_count})，title_sim={title_sim:.2f}，强度={strength}{comp_suffix}"
        else:
            comp_suffix = "有竞争" if has_competition else "无竞争"
            reason = f"Top{position} 有合作者匹配({coauthor_count})，title_sim={title_sim:.2f}，{comp_suffix}"

    return subtype, reason, details


def process_records(
    df: pd.DataFrame,
    threshold: float,
) -> dict:
    """处理所有记录"""

    # 按 unass_record 分组
    grouped = df.groupby("unass_record")

    results = {}

    for unass_record, group in tqdm(grouped, desc="处理记录"):
        # 对每个候选作者，计算多个 title_sim 指标
        candidate_stats = (
            group.groupby("candidate_aid")
            .agg(
                {
                    "title_sim": [
                        "max",
                        lambda x: compute_topk_avg(list(x), 3),
                        lambda x: compute_topk_avg(list(x), 5),
                        lambda x: compute_topk_avg(list(x), 10),
                    ],
                    "coauthor_count": "max",
                    "coauthor_org_match_count": "max",
                    "org_sim": "max",
                    "venue_sim": "max",
                }
            )
            .reset_index()
        )

        # 重命名列
        candidate_stats.columns = [
            "aid",
            "title_sim_max",
            "title_sim_top3_avg",
            "title_sim_top5_avg",
            "title_sim_top10_avg",
            "coauthor_count",
            "coauthor_org_match_count",
            "org_sim",
            "venue_sim",
        ]

        # 向后兼容：保留 title_sim 作为 title_sim_max 的别名
        candidate_stats["title_sim"] = candidate_stats["title_sim_max"]

        # 按阈值筛选（使用 title_sim_max）
        filtered = candidate_stats[candidate_stats["title_sim_max"] >= threshold]

        # 按相似度降序排列
        candidates = filtered.sort_values("title_sim_max", ascending=False).to_dict(
            "records"
        )

        # 子类型分类
        subtype, reason, details = classify_record(candidates)

        # 向后兼容：推导难度等级
        difficulty = subtype_to_difficulty(subtype)

        results[unass_record] = {
            "subtype": subtype,
            "difficulty": difficulty,  # 向后兼容
            "reason": reason,
            "candidates": candidates,
            "subtype_details": details,
            # 向后兼容字段
            "l1_subtype": subtype if difficulty == "L1" else None,
            "l1_details": details if difficulty == "L1" else None,
        }

    return results


def print_statistics(results: dict) -> None:
    """打印筛选统计信息"""
    total_records = len(results)
    total_candidates = sum(len(r["candidates"]) for r in results.values())

    # 子类型统计
    subtype_counts: dict[str, int] = {}
    for r in results.values():
        subtype = r.get("subtype", "NONE")
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

    # 难度统计（向后兼容）
    difficulty_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    for r in results.values():
        difficulty = r.get("difficulty", "L0")
        if difficulty in difficulty_counts:
            difficulty_counts[difficulty] += 1

    logger.info("=" * 60)
    logger.info("筛选统计:")
    logger.info(f"  待消歧记录数: {total_records}")
    logger.info(f"  筛选后候选总数: {total_candidates:,}")
    logger.info(
        f"  无候选的记录数: {sum(1 for r in results.values() if not r['candidates'])}"
    )

    # 子类型分布
    logger.info("-" * 60)
    logger.info("子类型分布:")
    for subtype in sorted(subtype_counts.keys()):
        count = subtype_counts[subtype]
        logger.info(f"  {subtype}: {count:,} ({count / total_records:.1%})")

    # 难度分布（向后兼容）
    logger.info("-" * 60)
    logger.info("难度分布 (向后兼容):")
    difficulty_desc = {
        "L0": "无候选作者 (NONE)",
        "L1": "有合作者匹配 (P1/P2/P3)",
        "L2": "高相似度无合作者 (P0-High/Medium)",
        "L3": "低相似度无合作者 (P0-Low)",
    }
    for level in ["L0", "L1", "L2", "L3"]:
        count = difficulty_counts[level]
        logger.info(
            f"  {level}: {count:,} ({count / total_records:.1%}) - {difficulty_desc[level]}"
        )

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="基于相似度数据筛选候选作者")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["train", "valid", "test"],
        default="valid",
        help="数据集模式（默认: valid）",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="输入文件路径（默认根据 mode 自动选择）",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"标题相似度阈值，默认: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出文件路径（默认根据 mode 自动选择）",
    )
    args = parser.parse_args()

    # 根据模式设置默认路径
    input_path = args.input or f"results/{args.mode}_unass_sim_full.csv"
    output_path_str = args.output or f"results/{args.mode}_filtered_candidates.json"

    logger.info(f"模式: {args.mode}")
    logger.info(f"读取数据: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"总记录数: {len(df):,}")

    logger.info(f"筛选阈值: {args.threshold}")
    results = process_records(df, args.threshold)

    print_statistics(results)

    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
