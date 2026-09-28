# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from kvpress.presses.scorer_press import ScorerPress


class KVzapConfig(PretrainedConfig):
    model_type: str = "kvzap"
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    n_modules: int


class KVzapModel(PreTrainedModel):
    config_class = KVzapConfig  # type: ignore[assignment]

    def __init__(self, config):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        if config.hidden_dim is None:
            # Linear model
            self.layers = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            # 2-layer MLP model
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    def forward(self, x):
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)


@dataclass
class KVzapPress(ScorerPress):
    """
    KVzap (https://arxiv.org/abs/2601.07891) is a fast approximation of KVzip.
    It applies a lightweight surrogate model to the hidden states to predict importance scores.
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    
    # 使用 init=False 防止在实例化时被要求传入这些参数
    kvzap_model_name: Optional[str] = field(default=None, init=False)
    kvzap_model: Optional[KVzapModel] = field(default=None, init=False, repr=False)

    def post_init_from_model(self, model):
        """
        初始化钩子：根据主模型路径加载对应的 KVzap 权重
        """
        # 如果你希望根据主模型动态选择路径，可以使用 model.config.name_or_path
        # 这里使用你代码中硬编码的路径作为默认
        target_model_path = "/ssd1/tangziyao/KVzap-mlp-Llama-3.1-8B-Instruct"
        
        # 避免重复加载
        if self.kvzap_model is None or self.kvzap_model_name != target_model_path:
            print(f"--- [KVzap] 正在加载预训练权重: {target_model_path} ---")
            self.kvzap_model_name = target_model_path
            self.kvzap_model = KVzapModel.from_pretrained(target_model_path)
            self.kvzap_model.eval()

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        
        # --- 懒加载检查 (核心修复) ---
        if self.kvzap_model is None:
            self.post_init_from_model(None)

        # 获取对应层的 surrogate 模块
        kvzap_module = self.kvzap_model.layers[module.layer_idx]
        
        # --- 设备与精度同步 ---
        # 确保 KVzap 模型在正确的显卡上，并且 dtype 与 hidden_states 一致
        if kvzap_module[0].weight.device != hidden_states.device or kvzap_module[0].weight.dtype != hidden_states.dtype:
            self.kvzap_model = self.kvzap_model.to(device=hidden_states.device, dtype=hidden_states.dtype)
            kvzap_module = self.kvzap_model.layers[module.layer_idx]

        # 计算评分
        with torch.no_grad():
            # KVzap 通常对 hidden_states 进行变换，输出形状匹配 KV pair 数量的 scores
            scores = kvzap_module(hidden_states).transpose(1, 2)
            
        return scores