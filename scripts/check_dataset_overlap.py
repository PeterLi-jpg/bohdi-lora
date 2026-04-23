"""Verify HealthBench splits don't overlap before running the HealthBench-only
generalization experiment.

Sebastian asked whether training on HealthBench alone (no HealthBench Hard in
the training data) still helps on the HealthBench Hard holdout. That claim
only makes sense if the two splits have disjoint prompt_ids. If the same
prompt appears in both, "train on HealthBench only" accidentally leaks
Hard prompts into training.

This script answers:
  1. Are HealthBench and HealthBench Hard disjoint? (expected: yes)
  2. Are the 200 eval IDs in HealthBench Hard? (expected: yes, they're drawn from it)
  3. Are any eval IDs in HealthBench full? (expected: no, and relevant to the
     generalization experiment.)

Run:
    python scripts/check_dataset_overlap.py
"""

import argparse
import json
from pathlib import Path


def load_prompt_ids(path):
    """Return a set of prompt_ids from a HealthBench JSONL file."""
    ids = set()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.add(obj["prompt_id"])
    return ids


def load_eval_ids(path):
    """Load the fixed 200-prompt eval holdout. Accepts both list and dict schema."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data["prompt_ids"]
    return set(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthbench",
                        default="data/raw/healthbench.jsonl",
                        help="path to HealthBench full JSONL")
    parser.add_argument("--healthbench-hard",
                        default="data/raw/healthbench_hard.jsonl",
                        help="path to HealthBench Hard JSONL")
    parser.add_argument("--healthbench-consensus",
                        default="data/raw/healthbench_consensus.jsonl",
                        help="path to HealthBench Consensus JSONL")
    parser.add_argument("--eval-ids",
                        default="data/raw/hard_200_sample_ids.json",
                        help="path to the 200-prompt eval holdout file")
    args = parser.parse_args()

    for p in (args.healthbench, args.healthbench_hard, args.healthbench_consensus, args.eval_ids):
        if not Path(p).exists():
            raise SystemExit(f"missing file: {p}  (run scripts/download_data.py first)")

    full = load_prompt_ids(args.healthbench)
    hard = load_prompt_ids(args.healthbench_hard)
    consensus = load_prompt_ids(args.healthbench_consensus)
    eval_ids = load_eval_ids(args.eval_ids)

    full_hard_overlap = full & hard
    eval_in_hard = eval_ids & hard
    eval_in_full = eval_ids & full
    hard_is_subset = full_hard_overlap == hard
    consensus_full_overlap = full & consensus
    consensus_hard_overlap = hard & consensus
    consensus_is_subset_of_full = consensus_full_overlap == consensus

    print(f"HealthBench full:     {len(full):>6} prompts")
    print(f"HealthBench Hard:     {len(hard):>6} prompts")
    print(f"HealthBench Consensus:{len(consensus):>6} prompts")
    print(f"Eval holdout:         {len(eval_ids):>6} prompts")
    print()
    if hard_is_subset:
        print("full >= hard:         Hard is a subset of Full "
              "(generalization run must exclude Hard via --exclude-ids)")
    elif full_hard_overlap:
        print(f"full & hard:          {len(full_hard_overlap):>6} overlap "
              "(not a clean subset; inspect before running anything)")
    else:
        print("full & hard:          disjoint")

    if eval_in_hard == eval_ids:
        print("eval in hard:         all 200 eval IDs live in HealthBench Hard "
              "(expected)")
    else:
        print(f"eval in hard:         {len(eval_in_hard)} of {len(eval_ids)} "
              "eval IDs are in Hard (unexpected)")

    if eval_in_full:
        print(f"eval in full:         {len(eval_in_full)} of {len(eval_ids)} "
              "eval IDs appear in Full (HealthBench-only runs must exclude these)")
    else:
        print("eval in full:         no eval IDs in Full "
              "(HealthBench-only run safe without --exclude-ids)")

    if consensus_is_subset_of_full:
        print("consensus in full:    consensus is a subset of Full")
    elif consensus_full_overlap:
        print(f"consensus in full:    {len(consensus_full_overlap)} overlap "
              "(not a clean subset)")
    else:
        print("consensus in full:    disjoint")

    if consensus_hard_overlap:
        print(f"consensus in hard:    {len(consensus_hard_overlap)} overlap")
    else:
        print("consensus in hard:    disjoint")

    print()
    if hard_is_subset or eval_in_full:
        print("RECOMMENDATION for HealthBench-only generalization runs:")
        print("  --datasets healthbench \\")
        print("  --exclude-ids data/raw/healthbench_hard.jsonl "
              "data/raw/hard_200_sample_ids.json")
        print()
        raise SystemExit(
            "overlap present — Hard and/or eval IDs live inside Full. "
            "Not a bug; but a generalization run that trains on `--datasets "
            "healthbench` alone will leak Hard prompts unless excluded. "
            "See the recommendation above and re-run with --exclude-ids."
        )

    if eval_in_hard != eval_ids:
        missing = eval_ids - eval_in_hard
        print(f"WARN: {len(missing)} eval IDs are not in HealthBench Hard. "
              "The holdout should be drawn from Hard; inspect.")

    print("overlap check passed")


if __name__ == "__main__":
    main()
