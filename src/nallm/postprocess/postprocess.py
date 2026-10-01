"""后处理主流程编排

PostProcessor 类负责：
1. 加载模型和特征数据
2. 读取消歧结果 CSV
3. 逐条应用规则 A/B
4. 保存后处理结果
"""

import csv
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nallm.utils.io import safe_write_csv

from nallm.postprocess.config import THETA_A, THETA_B, MODEL_PATH
from nallm.postprocess.feature_compute import FeatureLookup
from nallm.postprocess.model import DisambiguationModel
from nallm.postprocess.rules import (
    apply_rules,
    compute_candidates_proba,
)

logger = logging.getLogger(__name__)


def _read_results_csv(path: Path) -> pd.DataFrame:
    """读取消歧结果 CSV

    使用 csv.DictReader 读取，自动处理 reasoning 含逗号的情况。
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    return pd.DataFrame(rows, columns=fieldnames)


class PostProcessor:
    """模型辅助后处理器"""

    def __init__(
        self,
        theta_a: float = THETA_A,
        theta_b: float = THETA_B,
        model_path: Path = MODEL_PATH,
    ):
        self.theta_a = theta_a
        self.theta_b = theta_b
        self.model_path = model_path
        self.model = DisambiguationModel()
        self.feature_lookup = FeatureLookup()

    def initialize(
        self,
        features_jsonl: Path,
    ) -> None:
        """初始化模型和特征数据

        Args:
            features_jsonl: 预计算的特征 JSONL 文件路径
        """
        logger.info("加载后处理模型...")
        self.model.load(self.model_path)

        # 加载特征数据
        self.feature_lookup.load(features_jsonl)

    def check_candidate_consistency(
        self,
        df: pd.DataFrame,
        strict: bool = True,
    ) -> list[tuple[str, int, int, str, str]]:
        """校验每条记录的特征候选数与 CSV num_candidates 一致

        防特征-候选池脱节:特征文件与 filter 候选不一致会导致该 record
        候选 proba=0 被静默跳过(规则 B 不触发)。若 LLM 阶段 filter 给了
        候选而特征文件无候选,应立即暴露,而不是产出错误结果后靠人工回溯。

        Args:
            df: 消歧结果 DataFrame(须含 unass_record/num_candidates/subtype/strategy 列)
            strict: True 时不一致立即抛错;False 时降级为警告(兼容旧特征文件)

        Returns:
            不一致记录列表 [(unass_record, num_candidates, 特征文件候选数, subtype, strategy)]
        """
        if "num_candidates" not in df.columns:
            logger.warning("CSV 无 num_candidates 列，跳过候选一致性校验")
            return []

        mismatches: list[tuple[str, int, int, str, str]] = []
        for row in df.itertuples(index=False):
            # 跳过 error 记录(LLM 调用失败,候选数无意义)
            if str(getattr(row, "predicted_author_id", "")) == "error":
                continue
            try:
                ncand = int(getattr(row, "num_candidates"))
            except (ValueError, TypeError):
                ncand = -1  # 缺失/异常值,视为不一致
            feat_c = self.feature_lookup.candidate_count(row.unass_record)
            if ncand != feat_c:
                mismatches.append(
                    (
                        row.unass_record,
                        ncand,
                        feat_c,
                        str(getattr(row, "subtype", "")),
                        str(getattr(row, "strategy", "")),
                    )
                )

        if mismatches:
            msg = (
                f"候选一致性校验失败: {len(mismatches)}/{len(df)} 条记录脱节"
                f"(num_candidates != 特征文件候选数)\n"
                f"前 10 条: {mismatches[:10]}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
        else:
            logger.info(
                f"候选一致性校验通过: {len(df)} 条记录 "
                "num_candidates 均与特征文件候选数一致"
            )

        return mismatches

    def process_results(
        self,
        results_csv: Path,
        output_csv: Path | None = None,
        strict_candidate_check: bool = True,
    ) -> pd.DataFrame:
        """对消歧结果应用后处理

        Args:
            results_csv: 消歧结果 CSV 路径
            output_csv: 输出路径，默认在原文件同目录添加 _postprocessed 后缀
            strict_candidate_check: 候选一致性校验是否严格(不一致即报错)

        Returns:
            后处理后的 DataFrame
        """
        if output_csv is None:
            output_csv = results_csv.with_name(
                results_csv.stem + "_postprocessed" + results_csv.suffix
            )

        # 读取结果
        logger.info(f"读取消歧结果: {results_csv}")
        df = _read_results_csv(results_csv)
        logger.info(f"共 {len(df)} 条记录")

        # 候选一致性校验:防止特征文件与候选池脱节、proba=0 被静默跳过
        self.check_candidate_consistency(df, strict=strict_candidate_check)

        # 统计变量
        rule_a_count = 0
        rule_b_count = 0
        skipped_count = 0

        # 新增列
        model_probas = []
        rules_applied = []
        original_aids = []
        modified_aids = []

        for row in tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="后处理",
        ):
            unass_record = row.unass_record
            predicted_aid = str(row.predicted_author_id)
            subtype = row.subtype
            strategy = getattr(row, "strategy", "")

            # 跳过无候选、自动分配、错误记录
            if (
                subtype == "NONE"
                or strategy == "auto"
                or predicted_aid == "error"
            ):
                skipped_count += 1
                model_probas.append(0.0)
                rules_applied.append(None)
                original_aids.append(predicted_aid)
                modified_aids.append(predicted_aid)
                continue

            # 查表获取候选概率并应用规则
            candidates_proba = compute_candidates_proba(
                unass_record, self.feature_lookup, self.model
            )

            result = apply_rules(
                predicted_aid, candidates_proba, self.theta_a, self.theta_b
            )

            model_probas.append(result.model_proba)
            rules_applied.append(result.rule_applied)
            original_aids.append(result.original_aid)
            modified_aids.append(result.modified_aid)

            if result.rule_applied == "A":
                rule_a_count += 1
            elif result.rule_applied == "B":
                rule_b_count += 1

        # 更新预测结果列
        df["predicted_author_id"] = modified_aids
        df["model_proba"] = model_probas
        df["rule_applied"] = rules_applied
        df["original_predicted_author_id"] = original_aids

        # 保存结果
        safe_write_csv(df, output_csv)

        # 统计日志
        logger.info("=" * 50)
        logger.info("后处理完成")
        logger.info(f"  总记录数: {len(df)}")
        logger.info(f"  跳过记录: {skipped_count}")
        logger.info(f"  规则 A 触发: {rule_a_count} (LLM 分配 → none)")
        logger.info(f"  规则 B 触发: {rule_b_count} (none → 分配)")
        logger.info(f"  θ_a={self.theta_a}, θ_b={self.theta_b}")
        logger.info(f"  输出文件: {output_csv}")
        logger.info("=" * 50)

        return df


__all__ = ["PostProcessor"]
