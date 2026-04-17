# Known Issues

Triage of issues raised against this repo, with status for each.

Issue numbers below match GitHub issues on `PeterLi-jpg/bohdi-lora`.

## Fixed

### [#2] `format_example` batching bug — FIXED
TRL probes `formatting_func` on a single example first to decide whether it returns str or list[str]. The previous code unconditionally zipped `batch["messages"]` and `batch["response"]`, which on a single-example call silently zipped message dicts with *characters* of the response string — producing malformed training text. `format_example` now handles both shapes explicitly (`scripts/train_lora.py`).

### [#5] Silent grader parse failures — FIXED
`grade_trace` now returns a `parse_failures` count per trace. Both `filter_traces.py` and `eval_healthbench.py` surface the aggregate parse-failure rate in their summary output. Raw failed grader outputs are attached as `raw_parse_failures` for inspection.

### [#6, #7] Resume logic mixing settings — FIXED
`generate_traces.py --resume-from` now validates that existing rows in the target file were produced with the same `--model` and `--use-bodhi` settings. Mismatch aborts with a clear error. Override with `--force-resume` only when intentionally mixing is desired.

## Documented, deferred to discussion

### [#1] Brier / ECE do not measure model calibration — PARTIALLY ADDRESSED
The implementation uses the evaluator's rubric score as "confidence" and compares it against rubric outcomes from the same grading pass. That is grader-internal consistency, not model calibration. **Action taken**: the fields in eval output are renamed to `brier_grader_consistency` / `ece_grader_consistency` with an in-file comment, and the console summary now flags them (`brier*`/`ece*` with a footnote). **Still open for discussion**: whether to remove these metrics entirely or replace them with a model-derived confidence signal (logprob-based, explicit confidence output, or similar).

### [#3] Same grader for filtering and final evaluation — OPEN
`filter_traces.py` and `eval_healthbench.py` default to the same grader family (`Qwen/Qwen2.5-14B-Instruct-AWQ`). This couples training-data selection to the evaluator used for claimed gains. **Mitigation paths**:
- Use a distinct grader for final eval (simplest — change `--grader-model` in `slurm/eval_lora.sh`)
- Report final results under a second independent grader and compare
- Keep filter grader ≠ eval grader as policy

Needs team agreement before any ablation is blessed as final.

### [#4] Inconsistent filtering-score normalization — OPEN
Score formula is `sum_of_met_points / sum_of_positive_points`. Negative rubric items contribute to the numerator as penalties but not to the denominator, so a fixed `--min-score 0.4` threshold is not comparable across prompts with different penalty structure. Data selection is therefore partially a function of rubric geometry, not just response quality.

**Options to consider**:
- Normalize by `sum(|points|)` so the score lives on a symmetric scale
- Apply per-prompt z-scoring before the filter
- Require a minimum fraction of positive items met, independent of points

Needs a call on which normalization to use, and a re-derivation of the filter threshold.

## Process note

Fixes landing in `main` happen behind commits signed by the original authors. Deferred items should be resolved in a short methodology write-up before the paper eval is considered final.
