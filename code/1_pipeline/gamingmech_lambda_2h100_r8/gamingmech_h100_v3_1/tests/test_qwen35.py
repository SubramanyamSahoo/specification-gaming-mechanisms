"""Qwen3.5 support: hybrid layers (softmax attention and Gated DeltaNet), LoRA targets, head sites, Qwen-Scope SAEs."""
import types

import pytest
import torch

from gamingmech.config import MODEL_REGISTRY, Config, LoraSpec
from gamingmech.features import TopKSAE, load_sae
from gamingmech.integration import HYBRID
from gamingmech.mech import site_module
from gamingmech.models import LM

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM = 32, 4, 2, 1, 8     # softmax layers get fewer heads than linear ones


def tiny_qwen35():
    """Tiny random Qwen3.5 text model with the real model's layer pattern (three linear, then one softmax)."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    hybrid = {**HYBRID["qwen"], "layer_types": ["linear_attention"] * (LAYERS - 1) + ["full_attention"]}
    conf = Qwen3_5TextConfig(vocab_size=64, hidden_size=HIDDEN, intermediate_size=2 * HIDDEN, num_hidden_layers=LAYERS,
                             num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM,
                             max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0, **hybrid)
    torch.manual_seed(5)
    cfg = Config(); cfg.model_key = "qwen"; cfg.runtime.device = "cpu"
    lm = LM.__new__(LM); lm.cfg = cfg; lm.spec = MODEL_REGISTRY["qwen"]
    lm.device = torch.device("cpu"); lm.dtype = torch.float32; lm.model = Qwen3_5ForCausalLM(conf).eval()
    lm._is_peft = False; lm._unembed32 = None; lm.active = None
    for p in lm.model.parameters(): p.requires_grad_(False)
    return lm, conf


def test_registry_is_qwen35_9b_with_both_layer_kinds_targeted():
    spec = MODEL_REGISTRY["qwen"]
    assert spec.hf_id == "Qwen/Qwen3.5-9B" and spec.eot_token == "<|im_end|>"
    assert dict(spec.chat_kwargs) == {"enable_thinking": False} and dict(spec.thinking_kwargs) == {"enable_thinking": True}
    assert {"q_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"} <= set(spec.attn_types) <= set(spec.lora_targets)
    assert set(spec.mlp_types) <= set(spec.lora_targets) and "out_proj" in spec.residual_writers
    assert MODEL_REGISTRY["llama"].attn_types == ("q_proj", "k_proj", "v_proj", "o_proj")     # Llama unchanged


def test_head_sites_follow_each_layer_kind():
    lm, conf = tiny_qwen35()
    for layer, kind in enumerate(conf.layer_types):
        site, heads, width = lm.head_site(layer)
        if kind == "full_attention":
            assert site is lm.layers[layer].self_attn.o_proj and (heads, width) == (HEADS, HEAD_DIM)
            assert site_module(lm, "attn", layer) is lm.layers[layer].self_attn
        else:
            assert site is lm.layers[layer].linear_attn.out_proj
            assert (heads, width) == (conf.linear_num_value_heads, conf.linear_value_head_dim)
            assert site_module(lm, "attn", layer) is lm.layers[layer].linear_attn
        assert site.in_features == heads * width
    assert lm.n_heads == max(HEADS, conf.linear_num_value_heads)


def test_registry_lora_targets_reach_linear_and_softmax_layers():
    lm, conf = tiny_qwen35()
    lm.new_adapter("probe", LoraSpec(r=2, alpha=4, dropout=0.), seed=3)
    found = {(layer, mtype) for layer, mtype, _ in lm.lora_modules("probe")}
    for layer, kind in enumerate(conf.layer_types):
        names = ("q_proj", "k_proj", "v_proj", "o_proj") if kind == "full_attention" else ("in_proj_qkv", "in_proj_z", "out_proj")
        assert {(layer, n) for n in names + ("gate_proj", "up_proj", "down_proj")} <= found
    assert not any(mtype in ("in_proj_a", "in_proj_b", "conv1d") for _, mtype in found)


def test_head_attribution_marks_missing_heads(monkeypatch):
    """Softmax layers have fewer heads than linear ones here; the missing ones are NaN and never selected."""
    from gamingmech import patching
    lm, conf = tiny_qwen35()

    class Task:
        def __init__(self, *_a): pass

        def batch(self, lm_, _sl):
            ids = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 3]]); mask = torch.ones_like(ids)
            return ids, mask, None, None, None

        def metric(self, lm_, ids, mask, *_a):
            return lm_.hidden(ids, mask).last_hidden_state[:, -1].sum(-1)

    monkeypatch.setattr(patching, "PatchTask", Task)
    lm.using = lambda _name: __import__("contextlib").nullcontext()
    cfg = Config(); cfg.mech.attr_batch_size = 2
    attr = patching.head_attribution(lm, cfg, "a", "b", [{}, {}])
    assert attr.shape == (LAYERS, lm.n_heads)
    for layer, kind in enumerate(conf.layer_types):
        heads = lm.head_site(layer)[1]
        assert torch.isfinite(attr[layer, :heads]).all() and torch.isnan(attr[layer, heads:]).all()
    flat = attr.abs().flatten()
    real = torch.isfinite(flat).nonzero().flatten()
    top = real[flat[real].topk(min(64, real.numel())).indices].tolist()
    assert all(torch.isfinite(flat[i]) for i in top)


def test_topk_dictionary_keeps_k_features_and_subsets_after_the_top_k():
    torch.manual_seed(0)
    d, f, k = 8, 32, 4
    state = {"W_enc": torch.randn(f, d), "b_enc": torch.randn(f), "W_dec": torch.randn(d, f), "b_dec": torch.randn(d)}
    sae = TopKSAE(state, torch.device("cpu"), k)
    x = torch.randn(5, d)
    full = sae.encode(x)
    assert full.shape == (5, f) and ((full != 0).sum(-1) == k).all()
    idx = torch.tensor([0, 3, 7, 30])
    torch.testing.assert_close(sae.encode(x, idx), full[:, idx])
    torch.testing.assert_close(sae.decode(full), full @ state["W_dec"].T + state["b_dec"])
    assert sae.n_features == f


def test_qwen_scope_layout_loads_the_requested_layer_only(tmp_path):
    d, f = 8, 16
    for layer in (1, 11):
        torch.save({"W_enc": torch.full((f, d), float(layer)), "b_enc": torch.zeros(f), "W_dec": torch.zeros(d, f),
                    "b_dec": torch.zeros(d)}, tmp_path / f"layer{layer}.sae.pt")
    spec = types.SimpleNamespace(sae_repo=str(tmp_path), sae_file="layer{layer}.sae.pt", sae_trainer="unused",
                                 sae_kind="topk", sae_k=2)
    lm = types.SimpleNamespace(spec=spec, device=torch.device("cpu"))
    sae = load_sae(lm, Config(), 1)
    assert isinstance(sae, TopKSAE) and sae.k == 2 and float(sae.w_enc[0, 0]) == 1.0
    with pytest.raises(FileNotFoundError):
        load_sae(lm, Config(), 2)
