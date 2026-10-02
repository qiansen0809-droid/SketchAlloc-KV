"""Audit whether distinct LU-KV profiles actually change runtime budgets and masks.

Runs one held-out context under a small set of profiles, then fingerprints:
- exact per-layer/per-KV-head keep counts used by LUPress;
- the runtime masked-key index set left on the attention modules;
- the generated prediction.

This distinguishes "profiles did not take effect" from "profiles changed the
compressed cache but greedy generation stayed unchanged".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

import kvpress  # noqa: F401
from evaluation.cross_scene.evaluate_code_quality import (
    load_manifest,
    load_profile,
    load_samples,
    make_press,
)


def sha_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def mask_fingerprint(model) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    for layer_idx, layer in enumerate(model.model.layers):
        masked = getattr(layer.self_attn, "masked_key_indices", None)
        h.update(layer_idx.to_bytes(4, "little", signed=False))
        if masked is None:
            h.update(b"NONE")
            continue
        batch_idx, head_idx, seq_idx = masked
        for tensor in (batch_idx, head_idx, seq_idx):
            arr = tensor.detach().cpu().numpy().astype(np.int64, copy=False)
            h.update(arr.tobytes())
        total += int(seq_idx.numel())
    return h.hexdigest()[:16], total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", default="lcc")
    parser.add_argument("--profiles", default="official_lu,task_lcc")
    parser.add_argument("--compression", type=float, default=0.80)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    samples = load_samples(args.samples)
    candidates = [
        item for item in manifest["records"]
        if item["task"] == args.task and item["split"] == "heldout"
    ]
    if not candidates:
        raise ValueError(f"No heldout context for task {args.task}")
    record = candidates[0]
    context_id = record["context_id"]
    sample = samples[context_id]

    names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    pct = int(round(args.compression * 100))
    profile_arrays = {}
    shape = None
    for name in names:
        path = args.profiles_dir / f"{name}_ratio_{pct}.npy"
        profile = load_profile(path, shape)
        shape = profile.shape if shape is None else shape
        profile_arrays[name] = profile

    print("=== PROFILE ARRAY DIFFERENCES ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = np.abs(profile_arrays[a] - profile_arrays[b])
            print(
                f"{a} vs {b}: max_abs={diff.max():.6f}, "
                f"mean_abs={diff.mean():.6f}, changed_entries={np.sum(diff > 1e-12)}/{diff.size}"
            )

    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model_path,
        model_kwargs={
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
        },
        device_map=args.device_map,
        trust_remote_code=True,
    )
    pipe.model.eval()

    results = {}
    for name in names:
        press = make_press(
            profile_arrays[name],
            compression=args.compression,
            sink=args.sink,
            window=args.window,
        )
        output = pipe(
            sample["context"],
            question=sample["questions"][0],
            answer_prefix=sample["answer_prefix"],
            press=press,
            max_new_tokens=args.max_new_tokens,
            dataset_name="longbench",
            data_dir_name=args.task,
        )
        counts = press.last_keep_counts
        count_payload = {
            str(layer): [int(x) for x in values]
            for layer, values in sorted(counts.items())
        }
        count_array = np.asarray(
            [count_payload[str(layer)] for layer in sorted(counts)],
            dtype=np.int64,
        )
        mask_hash, total_pruned = mask_fingerprint(pipe.model)
        prediction = output["answer"]
        results[name] = {
            "count_hash": sha_json(count_payload),
            "count_array": count_array,
            "mask_hash": mask_hash,
            "total_pruned": total_pruned,
            "prediction": prediction,
        }
        print(f"\n=== {name} ===")
        print(f"context_id={context_id}")
        print(f"keep_count_hash={results[name]['count_hash']}")
        print(f"total_kept={int(count_array.sum())}")
        print(f"runtime_mask_hash={mask_hash}")
        print(f"total_pruned_mask_entries={total_pruned}")
        print(f"prediction={prediction!r}")

    print("\n=== PAIRWISE RUNTIME DIFFERENCES ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca = results[a]["count_array"]
            cb = results[b]["count_array"]
            print(
                f"{a} vs {b}: "
                f"heads_with_different_keep_count={int(np.sum(ca != cb))}/{ca.size}, "
                f"max_keep_count_delta={int(np.max(np.abs(ca - cb)))}, "
                f"mask_hash_equal={results[a]['mask_hash'] == results[b]['mask_hash']}, "
                f"prediction_equal={results[a]['prediction'] == results[b]['prediction']}"
            )


if __name__ == "__main__":
    main()
