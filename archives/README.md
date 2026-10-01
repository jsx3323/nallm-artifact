# Leaderboard submissions (predictions only)

Naming: `{split}_{model}_{stage}.json` — model ∈ deepseek (frozen main
line, DeepSeek-V3.2 snapshot) / luna (GPT-5.6-luna chain) / v41flash
(deepseek-v4.1-flash live rebuild); stage ∈ llm_only (pipeline minus
stage 6) / postprocessed (stage 6 included). Exception:
`test_featonly_argmax_theta` is the no-LLM ablation (19-feature LightGBM
argmax, borrowed θ_a=0.14).

All scores are from the official WhoIsWho leaderboard channel
(https://www.biendata.xyz/competition/whoiswho2/), identical evaluation
for our submissions and RND-all's.

| File | Chain / stage | Official score (wF1) |
|---|---|---|
| test_deepseek_postprocessed.json | main line, ③+⑥ | **0.93951** (wP .9299 / wR .9493) — first place |
| test_featonly_argmax_theta.json | feature-only ablation | 0.93655 (wP .9268 / wR .9465) |
| test_v41flash_postprocessed.json | live rebuild, ③+⑥ | 0.93613 (wP .9272 / wR .9453) |
| test_luna_postprocessed.json | luna, ③+⑥ | 0.93707 |
| test_deepseek_llm_only.json | main line, ③ only | 0.92811 (wP .9148 / wR .9418) |
| test_luna_llm_only.json | luna, ③ only | 0.92339 |
| test_v41flash_llm_only.json | live rebuild, ③ only | not submitted (retained for agreement analysis: 95.65% to main-line ③, pipeline-level 99.01% to luna+⑥) |

Reference point: RND-all, the strongest task-designed system and the
benchmark paper's method, scores 0.93520 on the same channel.

Prediction format: `{record_id: [author_id_or_None]}` — no dataset
content is included.
