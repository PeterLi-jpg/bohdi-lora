"""Compare per-example scores between two graders on the same eval outputs."""

import argparse
import json


def load_scores(path):
    with open(path) as handle:
        payload = json.load(handle)
    runs = payload.get("grader_runs", [])
    if len(runs) < 2:
        raise SystemExit("need at least two grader_runs in the eval JSON")
    return runs[0], runs[1]


def rank(values):
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2 + 1
        for _, idx in ordered[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def pearson(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    return num / (den_x * den_y) if den_x and den_y else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json")
    parser.add_argument("--disagreement-threshold", type=float, default=0.3)
    args = parser.parse_args()

    first, second = load_scores(args.eval_json)
    second_scores = {row["prompt_id"]: row["score"] for row in second["results"]}

    prompt_ids = []
    x = []
    y = []
    disagreements = []
    for row in first["results"]:
        pid = row["prompt_id"]
        if pid not in second_scores:
            continue
        score_a = row["score"]
        score_b = second_scores[pid]
        prompt_ids.append(pid)
        x.append(score_a)
        y.append(score_b)
        if abs(score_a - score_b) > args.disagreement_threshold:
            disagreements.append({"prompt_id": pid, "score_a": score_a, "score_b": score_b})

    rho = pearson(rank(x), rank(y))
    print(f"{first['grader_model']} vs {second['grader_model']}: Spearman rho = {rho:.4f}")
    print(f"Disagreements > {args.disagreement_threshold}: {len(disagreements)}")
    for item in disagreements[:10]:
        print(
            f"  {item['prompt_id']}: {item['score_a']:.3f} vs {item['score_b']:.3f}"
        )


if __name__ == "__main__":
    main()
