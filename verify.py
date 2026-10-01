"""v2.3.2 复现一致率验证

比对 outputs/predict_test_v2.3.2_repro.json 与 941c837 原版
predict_test_v2.3.2.json 的决策一致率(复现报告同口径:
差异条数 = 仅A分配 + 仅B分配 + 交集内归属不同,分母 14479)。

期望:53 条差异 → 99.6340%
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
ROOT = Path(__file__).parent
TOTAL_RECORDS = 14479


def _baseline_path() -> Path:
    """对照基线:优先 outputs/ 原名(工作套件);工件仓内为随仓的
    archives/test_deepseek_postprocessed.json(与原版逐字节相同)。"""
    for p in (
        OUT / "predict_test_v2.3.2_orig.json",
        ROOT / "archives" / "test_deepseek_postprocessed.json",
    ):
        if p.exists():
            return p
    raise FileNotFoundError(
        "baseline not found: outputs/predict_test_v2.3.2_orig.json or "
        "archives/test_deepseek_postprocessed.json"
    )


def load_paper_map(path: Path) -> dict[str, str]:
    with open(path) as f:
        d = json.load(f)
    return {paper: aid for aid, papers in d.items() for paper in papers}


def main() -> int:
    repro = load_paper_map(OUT / "predict_test_v2.3.2_repro.json")
    orig = load_paper_map(_baseline_path())

    ka, kb = set(repro), set(orig)
    inter = ka & kb
    same = sum(1 for p in inter if repro[p] == orig[p])
    only_repro, only_orig = len(ka - kb), len(kb - ka)
    diff_aid = len(inter) - same
    total_diff = only_repro + only_orig + diff_aid

    print(f"交集 {len(inter)} 篇,归属一致 {same} ({same/len(inter):.2%})")
    print(f"仅复现分配 {only_repro} | 仅原版分配 {only_orig} | 交集内作者不同 {diff_aid}")
    print(f"差异合计 {total_diff} 条 → 一致率 {1 - total_diff/TOTAL_RECORDS:.4%}")

    if total_diff == 53:
        print("✅ 与复现基准一致 (99.6340%)")
        return 0
    print(f"⚠️ 差异数 {total_diff} != 53,复现漂移,请检查物料")
    return 1


if __name__ == "__main__":
    sys.exit(main())
