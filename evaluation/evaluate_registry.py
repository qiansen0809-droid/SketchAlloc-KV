# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer

from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVzipPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    PyramidKVWrap,
    KVzapPress,
    LUPress,
)


# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    # "ruler": "simonjegou/ruler",
    "ruler": "/ssd1/tangziyao/datasets/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    # "infinitebench": "MaxJeblick/InfiniteBench",
    # "infinitebench": "/ssd2/tangziyao/tzy/datasets/InfiniteBench",
    "infinitebench": "/ssd2/tangziyao/tzy/datasets/myinf",
    # "longbench": "Xnhyacinth/LongBench",
    "longbench": "/ssd1/tangziyao/datasets/longbench",
    "longbench-e": "Xnhyacinth/LongBench",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
}


PRESS_REGISTRY = {
    "adakv_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "adakv_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=2e-2)),
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": ExpectedAttentionPress(),
    "expected_attention_e2": ExpectedAttentionPress(epsilon=2e-2),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvzip": KVzipPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "no_press": None,
    "pyramidkv_snapkv":PyramidKVWrap(SnapKVPress()),
    "pyramidkv_keydiff":PyramidKVWrap(KeyDiffPress()),
    "pyramidkv_ea":PyramidKVWrap(ExpectedAttentionPress()),
    "pyramidkv_ea_e2":PyramidKVWrap(ExpectedAttentionPress(epsilon=2e-2)),
    "adakv_keydiff":AdaKVPress(KeyDiffPress()),
    "lu_snapkv": LUPress(press=SnapKVPress(),sink=4, window=32),
    "lu_keydiff": LUPress(press=KeyDiffPress(),sink=4, window=1),
    "lu_ea": LUPress(press=ExpectedAttentionPress(epsilon=2e-2),sink=4, window=1),
}
