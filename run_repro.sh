#!/bin/bash
# v2.3.2 复现一键脚本(独立套件版,无外部依赖)
#
# 用法: bash run_repro.sh   (在仓库根目录)
# 期望: 与 941c837 原版决策一致率 99.6340%(53 条差异),退出码 0
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data outputs

# Release 附件为 gzip 压缩(.gz);已解压则跳过
for f in test_deepseek_llm_only.csv test_candidate_features.jsonl; do
    if [ -f "data/$f.gz" ] && [ ! -f "data/$f" ]; then
        echo "== 解压 data/$f.gz =="
        gunzip -k "data/$f.gz"
    fi
done

echo "== 1/3 后处理 (θ_a=0.14, θ_b=0.20) =="
uv run python scripts/postprocess_llm_first.py -m test \
    -i data/test_deepseek_llm_only.csv \
    -f data/test_candidate_features.jsonl \
    --model models/lgbm_arbiter_19f.txt \
    --theta-a 0.14 --theta-b 0.20 \
    -o outputs/test_full_results_repro_postprocessed.csv 2>&1 | tail -6

echo "== 2/3 转换提交格式 =="
uv run python scripts/convert_to_predict_format.py -m test \
    -i outputs/test_full_results_repro_postprocessed.csv \
    -o outputs/predict_test_v2.3.2_repro.json 2>&1 | tail -3

echo "== 3/3 一致率验证 (vs outputs/predict_test_v2.3.2_orig.json) =="
uv run python verify.py
