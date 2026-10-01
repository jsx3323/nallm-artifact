"""
将 run_llm_first.py 的全量结果转换成 predict_result_example.json 格式

用法:
    uv run python scripts/convert_to_predict_format.py
    uv run python scripts/convert_to_predict_format.py --mode test
"""

import argparse
import csv
import json
from pathlib import Path


def convert_results_to_predict_format(
    input_csv: Path,
    output_json: Path,
) -> dict:
    """将消歧结果转换为预测格式

    输出格式: {"author_id": ["paper_id", ...]} 用于 RNDeval.py

    Args:
        input_csv: 输入 CSV 文件路径（valid_full_results.csv）
        output_json: 输出 JSON 文件路径

    Returns:
        转换后的字典
    """
    # 读取 CSV
    results = {}
    total_records = 0
    none_count = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_records += 1
            unass_record = row["unass_record"]
            predicted_author_id = row["predicted_author_id"]

            # 跳过 "none" 记录
            if predicted_author_id == "none":
                none_count += 1
                continue

            # 去掉 order 后缀 (pid-order -> pid)
            paper_id = unass_record.split("-")[0]

            # 添加到结果
            if predicted_author_id not in results:
                results[predicted_author_id] = []
            results[predicted_author_id].append(paper_id)

    # 保存 JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("转换完成:")
    print(f"  输入: {input_csv}")
    print(f"  输出: {output_json}")
    print(f"  总记录数: {total_records}")
    print(f"  有效预测作者数: {len(results)}")
    print(f"  none 记录数: {none_count}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将消歧结果转换为预测格式")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["valid", "test"],
        default="valid",
        help="数据集模式（默认: valid）",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="输入 CSV 文件路径（默认根据 mode 自动选择）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认根据 mode 自动选择）",
    )
    args = parser.parse_args()

    # 根据模式设置默认路径
    input_csv = (
        Path(args.input)
        if args.input
        else Path(f"results/disambiguation/{args.mode}_full_results_postprocessed.csv")
    )
    output_json = (
        Path(args.output)
        if args.output
        else Path(f"results/disambiguation/{args.mode}_predict_results.json")
    )

    if not input_csv.exists():
        print(f"错误: 输入文件不存在 {input_csv}")
        return

    convert_results_to_predict_format(input_csv, output_json)


if __name__ == "__main__":
    main()
