import argparse
import json

import numpy as np
import pytest

from evaluation.cross_scene.analyze import analyze, load_manifest
from evaluation.cross_scene.prepare import format_row, prepare


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def test_prepare_keeps_independent_contexts_and_task_prefixes(tmp_path):
    rows = {
        "lcc": [
            {"context": f"def function_{i} ( x ) : return x", "input": "continue"}
            for i in range(4)
        ],
        "repobench-p": [
            {"context": f"class Project_{i} : pass", "input": "continue"}
            for i in range(4)
        ],
    }
    args = argparse.Namespace(
        model_path="fake", out_dir=tmp_path / "prepared",
        tasks="lcc,repobench-p", per_task=3,
        calibration_per_task=2, min_tokens=4, max_tokens=30,
        scan_limit=10, seed=42,
    )
    result = prepare(
        args, dataset_loader=lambda task: rows[task], tokenizer=FakeTokenizer()
    )
    assert len(result["records"]) == 6
    assert len({item["context_sha256"] for item in result["records"]}) == 6
    assert all(item["context_id"].startswith("context_") for item in result["records"])
    samples = [json.loads(line) for line in
               (args.out_dir / "profile_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert samples[0]["answer_prefix"] == "Next line of code:\n"
    assert format_row("multi_news", {"context": "News", "input": ""})[2] == "Summary:"
    with pytest.raises(FileExistsError):
        prepare(args, dataset_loader=lambda task: rows[task], tokenizer=FakeTokenizer())


def test_manifest_rejects_duplicate_context_even_with_different_ids(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": [
        {"context_id": "context_a", "task": "a", "split": "calibration", "context_sha256": "same"},
        {"context_id": "context_b", "task": "b", "split": "heldout", "context_sha256": "same"},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)


def test_cross_scene_analyze_uses_context_heldout_only(tmp_path):
    raw_root = tmp_path / "raw"
    records = []
    length = 80
    for task in ("lcc", "repobench-p"):
        for index in range(4):
            context_id = f"context_{task}_{index}"
            folder = raw_root / context_id
            folder.mkdir(parents=True)
            scorer = np.broadcast_to(np.arange(length, dtype=np.float32), (1, 2, length)).copy()
            oracle = np.ones((1, 2, length), dtype=np.float32)
            preferred_head = 0 if task == "lcc" else 1
            oracle[0, preferred_head, 1:79] *= 10
            np.save(folder / "snapkv.npy", scorer)
            np.save(folder / "question_0.npy", oracle)
            records.append({
                "context_id": context_id,
                "task": task,
                "split": "calibration" if index < 2 else "heldout",
                "context_sha256": f"{task}-{index}",
            })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    official = np.full((99, 1, 2), 0.5)
    official_path = tmp_path / "official.npy"
    np.save(official_path, official)
    args = argparse.Namespace(
        manifest=manifest, raw_root=raw_root, official_profile=official_path,
        output_dir=tmp_path / "out", method="snapkv", sink=1, window=1,
        compression=0.5, threshold=99, max_budget_gap_fraction=0.02,
    )
    result = analyze(args)
    assert result["counts"]["lcc"] == {"calibration": 2, "heldout": 2}
    assert result["counts"]["repobench-p"] == {"calibration": 2, "heldout": 2}
    assert result["matrix"]["lcc"]["task_lcc"]["mean_proxy_loss"] < (
        result["matrix"]["lcc"]["task_repobench-p"]["mean_proxy_loss"]
    )
    assert result["matrix"]["repobench-p"]["task_repobench-p"]["mean_proxy_loss"] < (
        result["matrix"]["repobench-p"]["task_lcc"]["mean_proxy_loss"]
    )
    assert (args.output_dir / "transfer_results.json").is_file()
