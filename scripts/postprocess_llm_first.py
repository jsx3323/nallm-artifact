"""模型辅助后处理脚本

对消歧结果应用 LightGBM 模型辅助后处理：
- 规则 A: LLM 分配 + 模型概率 < θ_a → 改为 none（减少误判）
- 规则 B: LLM 返回 none + 模型概率 ≥ θ_b → 分配候选（减少漏判）

特征数据从预计算的 JSONL 文件查表获取，无需实时计算。

用法:
    uv run python scripts/postprocess_llm_first.py --mode valid
    uv run python scripts/postprocess_llm_first.py --mode valid --input results/disambiguation/valid_full_results.csv
    uv run python scripts/postprocess_llm_first.py --mode test --theta-a 0.15 --theta-b 0.60
"""

import argparse
import logging
from pathlib import Path

from nallm.core.config import RESULTS_DIR
from nallm.postprocess import PostProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="模型辅助后处理")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["valid", "test"],
        default="valid",
        help="数据模式",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="输入消歧结果 CSV 路径（默认: results/disambiguation/{mode}_full_results.csv）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出路径（默认: 输入文件名加 _postprocessed 后缀）",
    )
    parser.add_argument(
        "--features",
        "-f",
        type=str,
        default=None,
        help="特征 JSONL 路径（默认: results/{mode}_features.jsonl）",
    )
    parser.add_argument(
        "--theta-a",
        type=float,
        default=0.12,
        help="规则 A 阈值: LLM 分配 + model_prob < θ_a → none（默认: 0.12）",
    )
    parser.add_argument(
        "--theta-b",
        type=float,
        default=0.50,
        help="规则 B 阈值: LLM 返回 none + model_prob ≥ θ_b → 分配（默认: 0.50）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型文件路径（默认: models/lgbm_arbiter_19f.txt）",
    )
    parser.add_argument(
        "--relax-candidate-check",
        action="store_true",
        help="候选一致性校验降级为警告（不中断），用于兼容旧特征文件",
    )
    args = parser.parse_args()

    # 确定输入路径
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = RESULTS_DIR / "disambiguation" / f"{args.mode}_full_results.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(
            input_path.stem + "_postprocessed" + input_path.suffix
        )

    # 确定特征文件路径
    if args.features:
        features_path = Path(args.features)
    else:
        features_path = RESULTS_DIR / f"{args.mode}_features.jsonl"

    if not features_path.exists():
        raise FileNotFoundError(
            f"特征文件不存在: {features_path}\n"
            f"请先运行: uv run python scripts/main_feat.py --mode {args.mode}"
        )

    # 确定模型路径
    model_path = Path(args.model) if args.model else None

    # 初始化后处理器
    logger.info("初始化后处理器...")
    postprocessor = PostProcessor(
        theta_a=args.theta_a,
        theta_b=args.theta_b,
        model_path=model_path,
    )
    postprocessor.initialize(features_path)

    # 处理结果
    logger.info(f"输入: {input_path}")
    logger.info(f"特征: {features_path}")
    logger.info(f"输出: {output_path}")
    logger.info(f"阈值: θ_a={args.theta_a}, θ_b={args.theta_b}")

    df = postprocessor.process_results(
        input_path,
        output_path,
        strict_candidate_check=not args.relax_candidate_check,
    )

    # 统计
    rule_a_count = (df["rule_applied"] == "A").sum()
    rule_b_count = (df["rule_applied"] == "B").sum()
    no_rule_count = df["rule_applied"].isna().sum()

    logger.info("=" * 50)
    logger.info("后处理统计")
    logger.info(f"  总记录数: {len(df)}")
    logger.info(f"  规则 A 触发: {rule_a_count} (LLM 分配 → none)")
    logger.info(f"  规则 B 触发: {rule_b_count} (none → 分配)")
    logger.info(f"  无规则触发: {no_rule_count}")
    logger.info(f"  θ_a={args.theta_a}, θ_b={args.theta_b}")
    logger.info(f"  输出文件: {output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
