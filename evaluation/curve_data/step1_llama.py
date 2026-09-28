import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import shutil
import argparse
from dataclasses import dataclass, field
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= CLI argument parsing =================
parser = argparse.ArgumentParser(description="Extract Attention and EA scores for Llama")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cuda_device", type=str, default="0")
parser.add_argument("--max_new_tokens", type=int, default=64)
parser.add_argument("--answer_prefix", type=str, default="回答：")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--methods",
    type=str,
    default="snapkv,keydiff,ea",
    help="Comma-separated scorer statistics to record. Oracle value norms remain enabled.",
)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

@dataclass
class AnalyzeConfig:
    model_path: str = args.model_path
    dataset_path: str = args.dataset_path
    output_dir: str = args.output_dir
    max_new_tokens: int = args.max_new_tokens
    answer_prefix: str = args.answer_prefix
    seed: int = args.seed
    methods: str = args.methods
    use_value_norm_weighting: bool = True

    def method_enabled(self, name: str) -> bool:
        return name in {part.strip() for part in self.methods.split(",") if part.strip()}
    snapkv_window_size: int = 32
    snapkv_kernel_size: int = 7
    ea_future_positions: int = 512
    ea_n_sink: int = 4
    ea_use_covariance: bool = True
    ea_epsilon: float = 0.02
    dummy_context: str = "人工智能在科学研究中的应用前景非常广阔。" * 50
    dummy_questions: List[str] = field(default_factory=lambda: ["这句话的主旨是什么？"])

# ================= Helper functions =================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_local(q, k, cos, sin):
    if cos.device != q.device:
        cos = cos.to(q.device)
        sin = sin.to(q.device)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class SampleData:
    context: str
    questions: List[str]
    sample_id: str = "0"
    task: str = "default"

# ================= Core Recorder =================
class Recorder:
    def __init__(self, config: AnalyzeConfig):
        self.config = config
        self.context_len = 0
        self.layer_num = 0
        self.kv_heads = 0
        self.max_attn_table = None 
        self.key_diff_table = None 
        self.snapkv_table = None
        self.ea_table = None
        self.normalized_value_norm_table = None
        self.collecting_context_metrics = False 
        self.is_decoding = False
        self.rope_module = None

    def set_rope_module(self, rope_module):
        self.rope_module = rope_module

    def reset_for_new_question(self, context_len, num_layers, num_kv_heads, device):
        self.context_len = context_len
        self.layer_num = num_layers
        self.kv_heads = num_kv_heads
        self.max_attn_table = torch.zeros((num_layers, num_kv_heads, context_len), dtype=torch.float32, device=device)
        self.is_decoding = False

    def enable_context_analysis(self):
        self.collecting_context_metrics = True
        self.key_diff_table = {}
        self.snapkv_table = {}
        self.ea_table = {}
        if self.config.use_value_norm_weighting:
            self.normalized_value_norm_table = {}

    def disable_context_analysis(self):
        self.collecting_context_metrics = False

    def compute_and_store_value_norms(self, layer_idx, value_states, o_proj_weight, num_heads, num_kv_heads, head_dim):
        if self.normalized_value_norm_table is None: self.normalized_value_norm_table = {}
        if layer_idx in self.normalized_value_norm_table: return
        bsz, kv_heads, seq_len, _ = value_states.shape
        if seq_len > self.context_len:
            value_states = value_states[:, :, :self.context_len, :]
            seq_len = self.context_len
        group_size = num_heads // num_kv_heads
        v_expanded = value_states.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        v_expanded = v_expanded.reshape(bsz, num_heads, seq_len, head_dim)
        hidden_size = o_proj_weight.shape[0]
        w_o_reshaped = o_proj_weight.view(hidden_size, num_heads, head_dim).permute(1, 0, 2) 
        target_device = o_proj_weight.device
        v_expanded = v_expanded.to(target_device)
        projected = torch.einsum("bhld,hmd->bhlm", v_expanded.float(), w_o_reshaped.float())
        raw_norms = torch.norm(projected, p=2, dim=-1).squeeze(0)
        layer_sum = raw_norms.sum() + 1e-9
        normalized_norms = raw_norms / layer_sum
        self.normalized_value_norm_table[layer_idx] = normalized_norms.detach()

    def compute_keydiff(self, layer_idx, key_states):
        keys = key_states[:, :, :self.context_len, :].float() 
        keys_norm = F.normalize(keys, p=2, dim=-1)
        anchor = keys_norm.mean(dim=2, keepdim=True) 
        sim = F.cosine_similarity(keys_norm, anchor, dim=-1)
        metric = -sim.squeeze(0) 
        if self.key_diff_table is None: self.key_diff_table = {}
        self.key_diff_table[layer_idx] = metric.detach().cpu()

    def compute_snapkv(self, layer_idx, attn_weights, num_heads, num_kv_heads):
        seq_len = attn_weights.shape[-1]
        bsz = attn_weights.shape[0]
        window = self.config.snapkv_window_size
        kernel = self.config.snapkv_kernel_size
        if seq_len <= window:
            self.snapkv_table[layer_idx] = torch.ones((num_kv_heads, seq_len), dtype=torch.float32).cpu()
            return
        sub_attn = attn_weights[..., -window:, : -window] 
        scores = sub_attn.mean(dim=-2) 
        scores = F.max_pool1d(scores, kernel_size=kernel, padding=kernel // 2, stride=1)
        group_size = num_heads // num_kv_heads
        if group_size > 1:
            scores = scores.view(bsz, num_kv_heads, group_size, -1).mean(dim=2) 
        max_val = scores.max().item() if scores.numel() > 0 else 1.0
        padding = torch.full((bsz, num_kv_heads, window), max_val, device=scores.device, dtype=scores.dtype)
        final_scores = torch.cat([scores, padding], dim=-1) 
        self.snapkv_table[layer_idx] = final_scores.squeeze(0).detach().cpu()

    def compute_ea(self, layer_idx, module, hidden_states, key_states, value_states):
        if self.ea_table is None: self.ea_table = {}
        if self.rope_module is None: return
        n_sink = self.config.ea_n_sink
        bsz, q_len, _ = hidden_states.shape
        n_heads, d = module.config.num_attention_heads, module.head_dim
        h = hidden_states[:, n_sink :, :].float()
        Wq = module.q_proj.weight.to(dtype=torch.float32) 
        mean_h = torch.mean(h, dim=1, keepdim=True) 
        mu = torch.matmul(mean_h, Wq.T).squeeze(1).view(bsz, n_heads, d) 
        cov = None
        if self.config.ea_use_covariance:
            h_centered = h - mean_h
            cov_h = torch.matmul(h_centered.transpose(1, 2), h_centered) / h.shape[1]
            temp = torch.matmul(cov_h, Wq.T) 
            cov = torch.matmul(Wq, temp).view(bsz, n_heads, d, n_heads, d)
            cov = torch.diagonal(cov, dim1=1, dim2=3).permute(0, 3, 1, 2) 
        future_pos_ids = torch.arange(q_len, q_len + self.config.ea_future_positions, device=mu.device).unsqueeze(0)
        dummy_x = torch.zeros((1, future_pos_ids.shape[1], 1, d), device=mu.device, dtype=torch.float32)
        cos, sin = self.rope_module(dummy_x, future_pos_ids) 
        cos, sin = cos.to(device=mu.device, dtype=torch.float32), sin.to(device=mu.device, dtype=torch.float32)
        if cos.dim() == 4: cos, sin = cos.squeeze(1), sin.squeeze(1)
        Id = torch.eye(d, device=mu.device, dtype=torch.float32)
        P = torch.zeros((d, d), device=mu.device, dtype=torch.float32)
        P[d // 2 :, : d // 2] = torch.eye(d // 2, device=mu.device, dtype=torch.float32)
        P[: d // 2, d // 2 :] = -torch.eye(d // 2, device=mu.device, dtype=torch.float32)
        R_stack = cos.unsqueeze(-1) * Id + sin.unsqueeze(-1) * P 
        R_avg = R_stack.mean(dim=1) 
        mu_rot = torch.matmul(mu, R_avg.transpose(1, 2))
        if self.config.ea_use_covariance and cov is not None:
            R_avg_exp = R_avg.unsqueeze(1) 
            cov_rot = torch.matmul(R_avg_exp, torch.matmul(cov, R_avg_exp.transpose(2, 3)))
        else:
            cov_rot = None
        num_kv_heads = key_states.shape[1]
        num_kv_groups = n_heads // num_kv_heads
        key_states_f32 = key_states.to(device=mu.device, dtype=torch.float32)
        keys_rep = repeat_kv(key_states_f32, num_kv_groups)
        scores = torch.matmul(mu_rot.unsqueeze(2), keys_rep.transpose(2, 3)).squeeze(2) / math.sqrt(d)
        if cov_rot is not None:
            term = torch.einsum("bhld,bhde,bhle->bhl", keys_rep, cov_rot, keys_rep)
            scores += term / (2 * d)
        scores = F.softmax(scores, dim=-1)
        scores = scores.view(bsz, num_kv_heads, num_kv_groups, -1).mean(dim=2) 
        v_norm = value_states.to(device=mu.device, dtype=torch.float32).norm(dim=-1) 
        final_scores = (scores + self.config.ea_epsilon) * v_norm
        if n_sink > 0:
            max_val = final_scores.max()
            final_scores[:, :, :n_sink] = max_val
        self.ea_table[layer_idx] = final_scores.squeeze(0).detach().cpu()

    def update_attn(self, layer_idx, attn_weights):
        if attn_weights.shape[-1] < self.context_len: return
        ctx_attn = attn_weights[..., :self.context_len].float().squeeze(0).squeeze(1) 
        if self.config.use_value_norm_weighting and self.normalized_value_norm_table is not None:
            if layer_idx in self.normalized_value_norm_table:
                norm_vals = self.normalized_value_norm_table[layer_idx]
                if norm_vals.device != ctx_attn.device: norm_vals = norm_vals.to(ctx_attn.device)
                weighted_score = ctx_attn * norm_vals
            else: weighted_score = ctx_attn 
        else: weighted_score = ctx_attn 
        q_heads = weighted_score.shape[0]
        group_size = q_heads // self.kv_heads
        if group_size > 1:
            reshaped = weighted_score.view(self.kv_heads, group_size, self.context_len)
            kv_head_max, _ = reshaped.max(dim=1) 
        else: kv_head_max = weighted_score
        self.max_attn_table[layer_idx] = torch.maximum(self.max_attn_table[layer_idx], kv_head_max)

recorder = None 

# ================= Model attention wrapper =================
def custom_attn_forward_wrapper(layer_idx, original_forward, config_obj):
    def forward(self, hidden_states, *args, **kwargs):
        position_embeddings = kwargs.get('position_embeddings')
        attention_mask = kwargs.get('attention_mask')
        past_key_value = kwargs.get('past_key_value')
        cache_position = kwargs.get('cache_position')
        if position_embeddings is None and len(args) > 0: position_embeddings = args[0]
        if attention_mask is None and len(args) > 1: attention_mask = args[1]
        if position_embeddings is None: return original_forward(hidden_states, *args, **kwargs)

        num_heads = self.config.num_attention_heads
        num_key_value_heads = self.config.num_key_value_heads
        head_dim = getattr(self, "head_dim", self.config.hidden_size // num_heads)
        seq_len, bsz = hidden_states.shape[1], hidden_states.shape[0]

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_local(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

        if seq_len > 1 and recorder.collecting_context_metrics:
            if (
                config_obj.method_enabled("keydiff")
                and key_states.shape[2] >= recorder.context_len
            ):
                recorder.compute_keydiff(layer_idx, key_states)
            if config_obj.use_value_norm_weighting and hasattr(self, "o_proj"):
                recorder.compute_and_store_value_norms(
                    layer_idx,
                    value_states,
                    self.o_proj.weight,
                    num_heads,
                    num_key_value_heads,
                    head_dim,
                )
            if config_obj.method_enabled("ea"):
                recorder.compute_ea(
                    layer_idx,
                    self,
                    hidden_states,
                    key_states,
                    value_states,
                )

        key_states_rep = repeat_kv(key_states, num_heads // num_key_value_heads)
        value_states_rep = repeat_kv(value_states, num_heads // num_key_value_heads)

        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None: attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if (
            seq_len > 1
            and recorder.collecting_context_metrics
            and config_obj.method_enabled("snapkv")
        ):
            recorder.compute_snapkv(
                layer_idx,
                attn_weights,
                num_heads,
                num_key_value_heads,
            )
        if seq_len == 1 and recorder.is_decoding:
            recorder.update_attn(layer_idx, attn_weights.detach())

        attn_output = torch.matmul(attn_weights, value_states_rep).transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output), attn_weights
    return forward

def read_data(path, default_task):
    samples = []
    if not os.path.exists(path): return samples
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                qs = obj.get('questions', []) or ([obj.get('input')] if 'input' in obj else []) or (['Default question?'])
                samples.append(SampleData(context=obj.get('context', ""), questions=qs, sample_id=str(idx), task=obj.get('task', default_task)))
            except: continue
            if len(samples) >= 200: break 
    return samples

def process_samples(config_obj, samples):
    global recorder
    recorder = Recorder(config_obj)
    tokenizer = AutoTokenizer.from_pretrained(config_obj.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config_obj.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager")
    
    if hasattr(model.model, "rotary_emb"): recorder.set_rope_module(model.model.rotary_emb)

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.forward = custom_attn_forward_wrapper(i, layer.self_attn.forward, config_obj).__get__(layer.self_attn, layer.self_attn.__class__)

    num_layers, num_kv_heads = len(model.model.layers), model.config.num_key_value_heads

    for s_idx, sample in enumerate(samples):
        print(f"\nProcessing Sample {s_idx} [Task: {sample.task}]...")
        sample_dir = os.path.join(config_obj.output_dir, f"context_{sample.sample_id}")
        if os.path.exists(sample_dir): shutil.rmtree(sample_dir) 
        os.makedirs(sample_dir)

        ctx_tokens = tokenizer.encode(sample.context, add_special_tokens=True, return_tensors="pt").to(model.device)
        ctx_len = ctx_tokens.shape[1]
        
        recorder.reset_for_new_question(ctx_len, num_layers, num_kv_heads, model.device)
        recorder.enable_context_analysis()
        with torch.no_grad(): model(ctx_tokens)
        recorder.disable_context_analysis()
        


        if recorder.key_diff_table:
            np.save(os.path.join(sample_dir, "keydiff.npy"), torch.stack([recorder.key_diff_table[i] for i in range(num_layers)]).numpy())
            

        if recorder.snapkv_table:
            np.save(os.path.join(sample_dir, "snapkv.npy"), torch.stack([recorder.snapkv_table[i] for i in range(num_layers)]).numpy())
            

        if recorder.ea_table:
            np.save(os.path.join(sample_dir, "ea.npy"), torch.stack([recorder.ea_table[i] for i in range(num_layers)]).numpy())
        
        for q_idx, question in enumerate(sample.questions):
            suffix_tokens = tokenizer.encode(
                question + config_obj.answer_prefix,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
            input_ids = torch.cat([ctx_tokens, suffix_tokens], dim=1)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)
            
            recorder.reset_for_new_question(ctx_len, num_layers, num_kv_heads, model.device)
            recorder.is_decoding = True

            # Keep the upstream sampling protocol but make each context/question
            # replay reproducible across profiling runs.
            question_seed = config_obj.seed + s_idx * 1000 + q_idx
            torch.manual_seed(question_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(question_seed)

            with torch.no_grad():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config_obj.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            recorder.is_decoding = False
            np.save(os.path.join(sample_dir, f"question_{q_idx}.npy"), recorder.max_attn_table.cpu().numpy())

if __name__ == "__main__":
    conf = AnalyzeConfig()
    data_samples = read_data(conf.dataset_path, "default")
    if not data_samples:
        data_samples = [SampleData(context=conf.dummy_context, questions=conf.dummy_questions, sample_id="dummy_0", task="default")]
    process_samples(conf, data_samples)