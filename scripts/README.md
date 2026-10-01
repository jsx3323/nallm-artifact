# Script index

All entry points for the paper's experiments. Requirements per script:
**[D]** benchmark data snapshots under `data/` (see repository README — not
shipped), **[K]** LLM API credentials via environment, **[H]** historical
working copies / run outputs (internal provenance audits, kept for the
record), **[–]** nothing beyond the repo (dry-run / conversion).

| Script | Purpose | Needs | Paper anchor |
|---|---|---|---|
| `postprocess_llm_first.py` | One-bit arbitration (postprocessing): rules A/B over LLM output with LightGBM scores (θ_a=0.14, θ_b=0.20) | D | §4.3, §5.3.2 |
| `convert_to_predict_format.py` | Convert pipeline output to leaderboard submission JSON | – | §5.2 |
| `verify.py` (repo root) | Compare reproduction against the scored main-line output; exit 0 = hit | D | §5.6 |
| `filter_candidates.py` | Candidate recall (filter v2, t=0.55) + subtyping | D | §4.1–4.2 |
| `replay_postprocess_valid.py` | Valid-side arbitration replay: LLM-only vs +arbitration, feature argmax, θ grid, per-subtype accuracies | D | §5.3.1–5.3.3 |
| `ablation_features_only.py` | Feature-only ablation on valid (argmax / argmax+θ) | D | §5.3.3 |
| `feat_only_test_submission.py` | Feature-only test submission file (same rules as above) | D | §5.3.3 |
| `theta_transfer_valid.py` | Split-half θ transfer: quantifies threshold-selection leakage | D | §5.3.2 |
| `run_llm_first.py` | Full pipeline CLI (stages 2–5 end-to-end) | D, K | §4 |
| `run_llm_first_batch.py` | Same, via the batch endpoint (57% cheaper) | D, K | §5.5 |
| `backfill_llm_first.py` | Resume/backfill failed records of a partial run; merge | D, K | ops |
| `naive_prompt_ablation.py` | Naive single-prompt vs stratified prompting, paired McNemar | D, K | §5.3.1 |
| `eval_v41flash_valid.py` | v4.1-flash valid evaluation + cross-chain agreement | D, H | §5.3.3 |

Status markers are honest: **[H]** scripts reproduce internal audits that
assumed historical working copies; they are included for completeness of
the record, not as turnkey paths.
