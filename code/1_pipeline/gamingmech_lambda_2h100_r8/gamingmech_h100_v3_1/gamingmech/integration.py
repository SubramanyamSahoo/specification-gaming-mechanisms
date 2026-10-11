"""Offline tiny Qwen3.5 + Llama integration checks using the installed HF/PEFT implementations.
No pretrained weights or user token are needed. Passing is NOT an 8B/H100 benchmark.
"""
from __future__ import annotations
import contextlib
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from .config import Config, MODEL_REGISTRY, LoraSpec
from .models import LM
from .gen import response_logprob_batch
from .mech import capture, hooks, out_tensor, with_tensor


# The tiny Qwen3.5 puts one layer of each kind in a two layer stack: softmax attention, then Gated DeltaNet.
HYBRID = {'qwen': dict(layer_types=['full_attention', 'linear_attention'], linear_num_value_heads=4,
                       linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4)}


def run_checks(device: str = 'cpu', dtype: torch.dtype = torch.float32) -> dict:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    import peft
    results = {}
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith('cuda') else []):
        for family, C, M in [('qwen', Qwen3_5TextConfig, Qwen3_5ForCausalLM), ('llama', LlamaConfig, LlamaForCausalLM)]:
            torch.manual_seed(19)
            cfg = Config(); cfg.model_key = family; cfg.runtime.device = device
            cfg.runtime.dtype = str(dtype).split('.')[-1]; cfg.train.logprob_chunk_tokens = 3
            cfg.train.gradient_checkpointing = True
            conf = C(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                     max_position_embeddings=128, attention_dropout=0., use_cache=False,
                     bos_token_id=1, eos_token_id=2, pad_token_id=0, **HYBRID.get(family, {}))
            conf._attn_implementation = 'sdpa'
            lm = LM.__new__(LM); lm.cfg = cfg; lm.spec = MODEL_REGISTRY[family]
            lm.device = torch.device(device); lm.dtype = dtype; lm.model = M(conf).to(device=device, dtype=dtype).eval()
            lm.hf_id = 'tiny-random-' + family; lm.revision = 'local'; lm._is_peft = False
            lm._unembed32 = None; lm.active = None; lm.eot_id = 2; lm.stop_ids = [2]
            lm.tok = SimpleNamespace(pad_token_id=0)
            for p in lm.model.parameters(): p.requires_grad_(False)
            for layer in range(lm.n_layers):                 # head patching splits these inputs into heads
                site, heads, width = lm.head_site(layer)
                if site.in_features != heads * width: raise AssertionError('Head layout does not match its projection')
            seqs = [[1, 4, 6, 7, 2], [1, 8, 3, 2]]; starts = [3, 2]
            tol = 0.025 if dtype == torch.bfloat16 else 2e-5
            with torch.no_grad():
                baseline = response_logprob_batch(lm, seqs, starts)
                ids, mask = lm.pad_batch(seqs, 'right')
                raw = lm.model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                target = torch.stack([raw[i, s-1:len(row)-1].log_softmax(-1).gather(1,
                            torch.tensor(row[s:], device=device)[:, None]).sum() for i, (row, s) in enumerate(zip(seqs, starts))])
                torch.testing.assert_close(baseline, target, atol=tol, rtol=tol)
                a = capture(lm, seqs, [[True]*len(x) for x in seqs], ('last', 'mean_all'), 2)
                b = capture(lm, seqs, [[True]*len(x) for x in seqs], ('last', 'mean_all'), 1)
                for name in a: torch.testing.assert_close(a[name], b[name], atol=tol, rtol=tol)
            # Exercise the actual Steerer/cache entry point before any paid model training.
            # The old compute() callback bug fails this even for a zero intervention.
            from .steering import Steerer
            from .gen import last_token_logits
            import numpy as np
            class TinyProbe:
                tf_tail = []; tf_seqs = []
                forced_seqs = seqs
                def __init__(self): self.calls = 0
                def values(self, model, batch_size, pre_forced=None, pre_tf=None):
                    self.calls += 1
                    return last_token_logits(model, self.forced_seqs, [3, 4], batch_size, pre_forced).float().numpy()[:, 0]
            with tempfile.TemporaryDirectory() as td:
                cfg.runtime.project = td
                q = TinyProbe(); steering = Steerer(lm, q)
                baseline_values = steering.values(0, None, 2)
                cached_values = steering.values(0, None, 2)
                if q.calls != 1: raise AssertionError("Steering recomputed a cached baseline")
                zero_values = steering.values(0, torch.zeros(lm.d_model), 2)
                np.testing.assert_allclose(zero_values, baseline_values, atol=tol, rtol=tol)
                np.testing.assert_allclose(cached_values, baseline_values, atol=tol, rtol=tol)
                if q.calls != 2: raise AssertionError("Steering callback did not execute exactly once per cache miss")
            cfg.runtime.project = None
            # Exact modules, non-reentrant checkpointing, a genuine backward and optimizer update.
            targets = ['model.layers.0.self_attn.q_proj', 'model.layers.1.mlp.down_proj']
            if family in HYBRID: targets.append('model.layers.1.linear_attn.out_proj')
            lm.new_adapter('candidate', LoraSpec(r=4, alpha=8, dropout=0., exact_targets=targets), 23)
            opt = torch.optim.AdamW([p for p in lm.model.parameters() if p.requires_grad], lr=.01)
            lp = response_logprob_batch(lm, seqs, starts)
            loss = -torch.nn.functional.logsigmoid((lp[0]-lp[1])-(baseline[0]-baseline[1]))
            loss.backward()
            grads = [p.grad for p in lm.model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not any(float(g.float().abs().sum()) > 0 for g in grads):
                raise AssertionError('Missing nonzero LoRA gradient')
            if not all(torch.isfinite(g).all() for g in grads): raise AssertionError('Nonfinite gradient')
            opt.step(); opt.zero_grad(); lm.model.gradient_checkpointing_disable(); lm.model.eval()
            with torch.no_grad(), lm.using(None):
                torch.testing.assert_close(response_logprob_batch(lm, seqs, starts), baseline, atol=tol, rtol=tol)
            ids, mask = lm.pad_batch(seqs, 'left'); store = {}
            def grab(_m, _a, out): store['raw'] = out_tensor(out).detach().clone()
            with torch.no_grad(), lm.using('candidate'), hooks([(lm.layers[-1], grab, False)]):
                donor = lm.unembed(lm.hidden(ids, mask).last_hidden_state[:, -1])
            if 'raw' not in store: raise AssertionError('Final raw hook did not run')
            def patch(_m, _a, out): return with_tensor(out, store['raw'])
            with torch.no_grad(), lm.using(None), hooks([(lm.layers[-1], patch, False)]):
                patched = lm.unembed(lm.hidden(ids, mask).last_hidden_state[:, -1])
            if donor.numel() == 0: raise AssertionError('Empty donor comparison')
            torch.testing.assert_close(patched, donor, atol=tol, rtol=tol)
            with tempfile.TemporaryDirectory() as td:
                lm.save_adapter('candidate', Path(td)/'adapter')
                lm.drop_adapter('candidate'); lm.load_adapter('roundtrip', Path(td)/'adapter')
                with torch.no_grad(), lm.using('roundtrip'):
                    reloaded = lm.unembed(lm.hidden(ids, mask).last_hidden_state[:, -1])
                torch.testing.assert_close(reloaded, donor, atol=tol, rtol=tol)
                lm.drop_adapter('roundtrip')
                # Behaviour and mechanism stages load, drop and reload every condition by name.
                for _ in range(2):
                    lm.load_adapter('candidate', Path(td)/'adapter')
                    with torch.no_grad(), lm.using('candidate'):
                        again = lm.unembed(lm.hidden(ids, mask).last_hidden_state[:, -1])
                    torch.testing.assert_close(again, donor, atol=tol, rtol=tol)
                    lm.drop_adapter('candidate')
            results[family] = {'passed': True, 'device': device, 'dtype': str(dtype),
                               'checks': ['head_sites', 'completion_likelihood', 'padding', 'exact_lora_targets',
                                          'checkpointed_gradient', 'frozen_base', 'raw_final_donor', 'adapter_roundtrip', 'adapter_reload_same_name', 'steering_lazy_callback_and_zero_identity']}
            del lm, opt, raw, a, b, grads
            if device.startswith('cuda'): torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    import json
    print(json.dumps(run_checks(), indent=2))
