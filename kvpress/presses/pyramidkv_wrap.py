import logging
from dataclasses import dataclass
import torch
from torch import nn
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

@dataclass
class PyramidKVWrap(BasePress):
    """
    Wrapper implementation of PyramidKV.
    Can wrap any ScorerPress that provides a score() method.
    """
    press: ScorerPress  # the underlying scorer to wrap, e.g. SnapKVPress
    beta: int = 20
    window_size: int = 64

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "PyramidKVPress must wrap a ScorerPress"
        assert self.beta >= 1, "Beta must be >= 1"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def get_layer_budget(self, module: nn.Module, q_len: int) -> int:
        """
        Compute the per-layer KV budget following the pyramid allocation strategy.
        """
        # Compute n_kept: the number of tokens to retain for this layer
        max_capacity_prompt = self.window_size + q_len * (1 - self.compression_ratio)
        min_num = (max_capacity_prompt - self.window_size) / self.beta
        max_num = (max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (max_capacity_prompt - self.window_size) * 2 - max_num

        if not (q_len >= max_num >= min_num >= self.window_size):
            return round(q_len * (1 - self.compression_ratio))

        steps = (max_num - min_num) / (module.config.num_hidden_layers - 1)
        return round(max_num - module.layer_idx * steps)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # Delegate scoring to the wrapped press
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # Compute the per-layer budget using the pyramid strategy
        q_len = hidden_states.shape[1]
        n_kept = self.get_layer_budget(module, q_len)

        # Select the top-k tokens by score
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
