"""Additional pure-logic regression tests for key helper functions."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


for mod in ["torch", "torch_xla", "datasets", "trl", "peft", "transformers", "tqdm", "yaml"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train_lora as train_lora
from scripts.eval_ushape import compute_tertile_cutoffs, tier_of, summarize
from scripts.generate_traces import load_exclude_ids


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = " | ".join(f"{m['role']}:{m['content']}" for m in messages)
        return rendered


@pytest.fixture(autouse=True)
def patch_tokenizer():
    original = train_lora._tokenizer
    train_lora._tokenizer = DummyTokenizer()
    yield
    train_lora._tokenizer = original


def test_format_example_single_shape():
    batch = {
        "messages": [{"role": "user", "content": "hello"}],
        "response": "world",
    }
    text = train_lora.format_example(batch)
    assert text == "user:hello | assistant:world"


def test_format_example_batched_shape():
    batch = {
        "messages": [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        "response": ["x", "y"],
    }
    text = train_lora.format_example(batch)
    assert text == ["user:a | assistant:x", "user:b | assistant:y"]


def test_format_example_preserves_existing_conversation():
    batch = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
        ],
        "response": "answer",
    }
    text = train_lora.format_example(batch)
    assert text == "system:rules | user:question | assistant:answer"


def test_compute_tertile_cutoffs_basic():
    meta = {
        "a": {"pos_points": 1},
        "b": {"pos_points": 2},
        "c": {"pos_points": 3},
        "d": {"pos_points": 4},
        "e": {"pos_points": 5},
        "f": {"pos_points": 6},
    }
    q1, q2 = compute_tertile_cutoffs(meta)
    assert q1 < q2
    assert 1 <= q1 <= 6
    assert 1 <= q2 <= 6


def test_compute_tertile_cutoffs_restricted_subset():
    meta = {
        "a": {"pos_points": 1},
        "b": {"pos_points": 10},
        "c": {"pos_points": 11},
        "d": {"pos_points": 12},
    }
    q1, q2 = compute_tertile_cutoffs(meta, restrict_to={"b", "c", "d"})
    assert q1 >= 10
    assert q2 >= q1


def test_compute_tertile_cutoffs_all_same_scores():
    meta = {str(i): {"pos_points": 5} for i in range(6)}
    q1, q2 = compute_tertile_cutoffs(meta)
    assert q1 == 5
    assert q2 == 5


def test_tier_of_edges():
    assert tier_of(1, 2, 4) == "easy"
    assert tier_of(2, 2, 4) == "easy"
    assert tier_of(3, 2, 4) == "medium"
    assert tier_of(5, 2, 4) == "hard"


def test_summarize_empty_input():
    assert summarize([], fail_threshold=0.4) == {"n": 0}


def test_summarize_basic_stats():
    out = summarize([0.1, 0.4, 0.9], fail_threshold=0.4)
    assert out["n"] == 3
    assert out["mean"] == pytest.approx((0.1 + 0.4 + 0.9) / 3)
    assert out["fail_rate"] == pytest.approx(1 / 3)


def test_summarize_bootstrap_is_deterministic():
    rng = __import__("numpy").random.default_rng(123)
    out1 = summarize([0.1, 0.4, 0.9], fail_threshold=0.4, bootstrap=50, rng=rng)
    rng = __import__("numpy").random.default_rng(123)
    out2 = summarize([0.1, 0.4, 0.9], fail_threshold=0.4, bootstrap=50, rng=rng)
    assert out1["mean_ci"] == out2["mean_ci"]
    assert out1["fail_rate_ci"] == out2["fail_rate_ci"]


def test_load_exclude_ids_from_json(tmp_path):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"prompt_ids": ["a", "b"]}))
    assert load_exclude_ids(str(path)) == {"a", "b"}


def test_load_exclude_ids_from_jsonl(tmp_path):
    path = tmp_path / "ids.jsonl"
    path.write_text('{"prompt_id":"a"}\n{"prompt_id":"b"}\n')
    assert load_exclude_ids(str(path)) == {"a", "b"}


def test_load_exclude_ids_mixed_inputs(tmp_path):
    json_path = tmp_path / "ids.json"
    json_path.write_text(json.dumps(["a", "b"]))
    jsonl_path = tmp_path / "ids.jsonl"
    jsonl_path.write_text('{"prompt_id":"c"}\n')
    assert load_exclude_ids([str(json_path), str(jsonl_path)]) == {"a", "b", "c"}
