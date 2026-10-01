"""LightGBM 模型加载与推理"""

import logging
from pathlib import Path

import numpy as np

from nallm.postprocess.config import MODEL_PATH

logger = logging.getLogger(__name__)


class DisambiguationModel:
    """消歧概率预测模型

    封装 LightGBM Booster 的加载和推理。
    """

    def __init__(self):
        self._booster = None

    def load(self, path: Path | None = None) -> None:
        """加载 LightGBM 模型（原生 .txt 格式）

        Args:
            path: 模型文件路径，默认使用配置中的 MODEL_PATH
        """
        import lightgbm as lgb

        model_path = path or MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请先运行: uv run python scripts/train_lgbm.py"
            )

        self._booster = lgb.Booster(model_file=str(model_path))
        logger.info(f"模型加载完成: {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self._booster is not None

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """批量预测正样本概率

        Args:
            features: (n_samples, n_features) 特征矩阵

        Returns:
            (n_samples,) 正样本概率数组
        """
        if self._booster is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        raw = self._booster.predict(features)
        # LightGBM 二分类 predict 直接返回概率
        return np.asarray(raw, dtype=np.float64)

    @staticmethod
    def save_model(model, path: Path | None = None) -> None:
        """保存 LightGBM 模型

        Args:
            model: LGBMClassifier 或 Booster 对象
            path: 保存路径，默认使用 MODEL_PATH
        """
        save_path = path or MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 支持 LGBMClassifier 和 Booster 两种输入
        booster = model.booster_ if hasattr(model, "booster_") else model
        booster.save_model(str(save_path))
        logger.info(f"模型已保存: {save_path}")


__all__ = ["DisambiguationModel"]
