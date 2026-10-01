"""后处理配置

定义模型辅助后处理的阈值和路径常量。
"""

from pathlib import Path

# 规则 A: LLM 分配 + model_prob < THETA_A → 改为 none
# 规则 B: LLM 返回 none + model_prob >= THETA_B → 分配模型推荐候选
# 冻结主线阈值(论文口径;CLI 可覆盖,run_repro.sh 即显式传入同值)
THETA_A = 0.14
THETA_B = 0.20

# 模型路径
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lgbm_arbiter_19f.txt"

__all__ = ["THETA_A", "THETA_B", "MODEL_DIR", "MODEL_PATH"]
