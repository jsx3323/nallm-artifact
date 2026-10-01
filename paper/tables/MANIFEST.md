# 论文数据清单（MANIFEST）

论文每个数字的唯一取源。`paper/tables/` 只收正文引用的最终表；原始产物在 `archives/`
（{split}_{model}_{stage} 命名）；证据编号指论文作者的内部实验台账（未随本复现包分发）。
取源栏中的 `scripts/case_studies.py` 等分析脚本同属内部台账，未随本复现包分发；本仓库 `scripts/` 为实验入口。
**2026-09-30 接手审计：§号与表号按当前正文重核**——表序 contrast/subtypes/dispatch/promptgrid/leaderboard/ladder/nil/decomp/buckets/think/repro，图序 axis/pipeline/prompts/strata；旧号系早期草稿残留。同日：subtypes 与 dispatch 两个 float 交换定义顺序（编号随之互换），使其与首次引用顺序一致。
**2026-10-01**：§5.2.4（Reasoning mode）并入 §5.2.3、其表撤（数字改正文内联、引用改指 §5.2.3）——表序末段自此为 buckets/repro，repro 表号 11→10；「think」仅在 09-30 这行里作为历史记录保留。
**2026-10-01（提炼清理轮）**：撤两表——dispatch（内容可由表 2 + 表 3 的 Serves 列导出）与 repro（两行数字已在 §5.5 正文与附录 B）。表序自此为 **1 contrast／2 subtypes／3 promptgrid／4 leaderboard／5 ladder／6 nil／7 decomp／8 buckets**，共 8 表；图序不变（1 axis／2 pipeline／3 prompts／4 strata／5 drift／6 errdir／7–9 附录三图）。
**2026-10-01（系统角度检查轮）**：新增 `bypass_coverage.csv`（§4.1 两层的覆盖与判对率，取源 `scripts/case_studies.py` 新段——注意该两层的预测是策略自身输出、非 LLM 判定）；errdir 行图号订正 5→6。
**2026-10-01（图 4 / 表 6 换回主线）**：两表改由 `scripts/strata_nil_tables.py` 生成，口径＝`scripts/case_studies.py`（valid，泄露口径，θ_a 0.14/θ_b 0.20；③=存档主线快照，⑥=按论文两规则重放；分层排除 bypass 层与无候选层）。此前两表用 v4.1/luna 代用链；主线无 valid 后处理存档件（`archives/valid_deepseek_postprocessed.csv` 不存在），故 ⑥ 一律走重放。正控：llm_only wF1 0.9158、gate 0.9400（=\FullValid）。
**2026-10-01（§5.2.3 难案例与策略论证）**：新增表 `tab:uniquecases`（record 级难案例，无 CSV、取源 `scripts/case_studies.py` 独有对段）；新增宏 `\UniqueLlm`／`\UniqueLlmKept`／`\UniqueLlmLost`／`\UniqueFeat`。

**2026-10-01（撤历史分层表）**：撤 `postprocess_gains_stratified_valid.csv`（两仓同删）——其「无后处理」列系主线 ③ 快照，而「后处理」列口径不可复现（θ 网格内 P2 目标 .9312 最高只到 .8943、P1-Medium-Comp 恒 .7783 对目标 .8426），正文无引用；分层形状主张自此由 图 4／`stratified_valid_fig3.csv`（主线口径，证据 52）承载。

**2026-10-02（§5.2.2 保持账）**：新增 `llm_keep_valid.csv`——LLM 正确决策保持率 99.4%（11,555/11,628，丢失 11+62=73）。取源探针 `/tmp/probe_llm_keep.py`（随 /tmp 清失时按 findings 14.33 口径用 case_studies.py 的加载与重放逻辑重写）；11 复用 `\UniqueLlmLost`、62 复用 `\CaseBrokeAbstain`，与 `\CaseBroke`{=}73 同一集合拆分。

**2026-10-02（定稿对齐轮）**：三件退场——contrast 表（§1 组件对照，整表删）、promptgrid 表、§5.4 Cost 整节（token/吞吐数字全部退场，`cost_throughput.csv` 两仓同撤，金额/耗时台账留证据 38–43、50）；图 gate 决策表图删；图 4 轨迹图换表 `tab:drift`。表序自此为 **1 subtypes／2 leaderboard／3 ladder／4 nil／5 drift／6 decomp／7 buckets／8 uniquecases**（8 表）；图序 **1 pipeline／2 prompts／3 strata／4 errdir／5–7 附录三图**（7 图）；§5 四节（main／ablations／robustness／repro，repro 由 5.5 改 5.4）。表图行引用的编号已按此序刷新。

| 论文位置 | 数字/内容 | 数据文件 | 证据 |
|---|---|---|---|
| 摘要 / §1 / §5.1 / 表 2 | 0.93951；对 RND-all 差 0.43pp、对亚军 kingsundad 差 0.20pp（端值现算）；前九名 | leaderboard_test.csv | 1.1 |
| 摘要 / §1 / §5.2.2 | 仲裁增益与误差率降（主线 +1.14pp／15.9%，luna +1.37pp／17.9%）| ladder_test.csv（F1 端值，误差率由端值现算）| 7、61 |
| §5.1 / 表 3 | IUAD→…→NALLM 阶梯 | ladder_test.csv | 1.1、5、6 |
| §4.1 / 表 1 | 子类型 test 实测分布 | subtype_distribution_test.csv | 快照实测 |
| §4.1 | bypass 与 NONE 两层的覆盖比与判对率（7.1%、99.0%、96.1%） | bypass_coverage.csv | 本节，取源 scripts/case_studies.py 两层段 |
| §5 / §5.3 | 两模型等价性（一致率/F1 差/p） | model_equivalence.csv | 22–26、55 |
| §5.1 | 版本序列与限定 | version_series.csv | 1.1 附注 |
| §5.2.1 / 图 3 | 分层准确率（主线，仅③ 对 +⑥）；正文 span 96→76 取本图口径；生成器 `scripts/strata_nil_tables.py` | stratified_valid_fig3.csv | 52（2026-10-01 主线口径） |
| 摘要 / §1 / §4.2 / §5.2.1 | 朴素提示对比（3.60pp） | naive_prompt_ablation.csv | 62 |
| §4.2 / §5.2.2 | A/B 触发数（主线与 luna 两链）、规则精度、**rule B 判到非真值候选的条数（=错归属，2/260）**、P0-Low 双阈值门占比 | rules_summary.csv | 10、11 |
| §5.2.2 / 表 4 | NIL 带（主线 + luna 两链 P/R 对比）；生成器同图 4 | nil_band_valid.csv | 64（2026-10-01 主线行取代 v4.1 行） |
| §5.2.2 | LLM 正确决策保持账（11628 对→保持 11555＝99.4%；丢失 11+62） | llm_keep_valid.csv | 14.33 |
| §5.2.3 | 思考开关（四组，2026-10-01 改正文内联、表已撤）；首行按现存文件重算 92.5/89.0/12:5（旧登记 93.6/90.3/20:7 不成立）| thinking_onoff.csv | 15–17、49、53、54 |
| §5.2.3 / 表 6 | 机制（漏判 7→21→24） | thinking_mechanism.csv | 18 |
| §5.2.3 | feat-only 消融 + θa 网格 + test 仲裁（2026-10-01 重算：主线记录集 + 官方 wF1；新增 corpus 列） | feat_only_ablation.csv | 44、63（旧值为 luna 语料 + micro 口径，已废，见 findings 14.13） |
| §5.2.3 / 图 4 | 错误方向按证据层（弃权型／误归属型构成；LLM 与 feature-only 两行组） | error_direction_by_stratum.csv | 本节，取源 scripts/case_studies.py 分层段 |
| §5.2.3 / 表 7 | 置信桶分解 | confidence_buckets.csv | 60 |
| §5.2.3 / 表 8 | LLM 独有对（归属对而特征错 30／组合保住 19／撤回 11；反向 733）与三条案例 | 无 CSV，取源 scripts/case_studies.py「LLM 独有对」段 | 本节 |
| §5.2.3 / 表 5 | 三条跨分裂轨迹与漂移（LLM-only 0.9158→0.92811＝+1.23pp；feature-only 0.9632→0.93655；组合 0.9400→0.93951）；valid 端 | valid 端＝scripts/case_studies.py 三方分解段，test 端＝ladder_test.csv | 44、63、61 |
| §5.3 | 三链两阶段一致率（LLM-only 95.40/96.56；提交件 98.82 主线 vs luna、99.01 luna vs v4.1） | model_equivalence.csv | 24、25、57 |
| §5.3 | 候选截断 mc5/mc7 | candidate_truncation.csv | 27 |
| §5.4 / 附录 B；§5.1 | 三层复现口径（2026-10-01 表撤，数字留正文与附录 B）；bag 单 seed 波动 0.9408±0.0008（valid，5 seeds） | reproducibility_tiers.csv | 28、33、36、37；README bag5 节 |

## 口径速查

test 榜单 = 无偏；valid = 泄露口径（θ 在 valid 网格；选择泄露已量化 ≤0.001，证据 56）；
准确率仅用于 §5.2.3 配对与分层画像，其余一律 wF1；「仅 LLM/llm_only」= 管线 minus ⑥。
