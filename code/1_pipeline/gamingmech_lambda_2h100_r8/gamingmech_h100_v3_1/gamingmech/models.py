"""Model loading, chat serialisation and adapter management.

Training and evaluation both build token sequences through `encode_prompt` and
`encode_dialogue`, so the format the adapters learn is the format they are tested on.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .common import hf_login
from .config import Config, ConditionSpec, LoraSpec, ModelSpec

Messages = List[Dict[str, str]]


class LM:
    """A causal LM plus tokenizer, with helpers that hide model family differences."""

    def __init__(self, cfg: Config, spec: Optional[ModelSpec] = None, hf_id: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.spec = spec or cfg.spec
        self.hf_id = hf_id or self.spec.hf_id
        token = hf_login(cfg.runtime.hf_token_env)
        from .runtime import resolve_revision
        if cfg.runtime.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this profile; CPU fallback is forbidden")
        if cfg.runtime.strict_token and not token and cfg.runtime.device != "cpu":
            raise RuntimeError("Set HF_TOKEN before loading models")
        self.device = torch.device(cfg.runtime.device)
        self.dtype = getattr(torch, cfg.runtime.dtype)
        self.revision = resolve_revision(cfg, self.hf_id)
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision, token=token)
        if self.spec.pad_token and self.spec.pad_token in self.tok.get_vocab():
            self.tok.pad_token = self.spec.pad_token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, revision=self.revision, torch_dtype=self.dtype, token=token,
            attn_implementation=cfg.runtime.attention, low_cpu_mem_usage=True,
            device_map={"": str(self.device)}).eval()
        self.model.config.use_cache = False
        for p in self.model.parameters(): p.requires_grad_(False)
        self.eot_id = self.tok.convert_tokens_to_ids(self.spec.eot_token)
        if self.eot_id is None or self.eot_id == self.tok.unk_token_id:
            raise ValueError(f"end of turn token {self.spec.eot_token!r} is not in the vocabulary")
        self.stop_ids = sorted({self.eot_id} | ({self.tok.eos_token_id} - {None}))
        self.active: Optional[str] = None     # name of the active adapter, None for the base model
        self._is_peft = False
        self._unembed32: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ architecture
    @property
    def hf_model(self):
        return self.model.get_base_model() if self._is_peft else self.model

    @property
    def layers(self):
        return self.hf_model.model.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def d_model(self) -> int:
        return self.hf_model.config.hidden_size

    @property
    def n_heads(self) -> int:
        """Most heads in any layer; layers of a hybrid model can have fewer."""
        return max(self.head_site(layer)[1] for layer in range(self.n_layers))

    def head_site(self, layer: int):
        """(module whose input concatenates the per head outputs, number of heads, head width) for one layer.

        Softmax attention layers write through o_proj; Gated DeltaNet layers through out_proj, whose input is the
        gated, per head normalised value read out of each head's recurrent state.
        """
        block, c = self.layers[layer], self.hf_model.config
        if hasattr(block, "self_attn"):
            return (block.self_attn.o_proj, c.num_attention_heads,
                    getattr(c, "head_dim", None) or self.d_model // c.num_attention_heads)
        return block.linear_attn.out_proj, c.linear_num_value_heads, c.linear_value_head_dim

    def autocast(self):
        enabled = self.device.type == "cuda" and self.dtype in (torch.bfloat16, torch.float16)
        return torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=enabled)

    # ------------------------------------------------------------------ serialisation
    def render(self, messages: Messages, chat_kwargs: Optional[Sequence[Tuple[str, object]]] = None) -> str:
        """Chat template text ending at the start of the assistant turn."""
        kw = dict(self.spec.chat_kwargs if chat_kwargs is None else chat_kwargs)
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)

    def encode_prompt(self, messages: Messages, chat_kwargs: Optional[Sequence[Tuple[str, object]]] = None,
                      prefill: str = "") -> List[int]:
        """Prompt tokens; `prefill` is text the assistant turn is forced to start with."""
        ids = self.tok(self.render(messages, chat_kwargs), add_special_tokens=False)["input_ids"]
        return ids + (self.tok(prefill, add_special_tokens=False)["input_ids"] if prefill else [])

    def encode_response(self, response: str) -> List[int]:
        return self.tok(response, add_special_tokens=False)["input_ids"] + [self.eot_id]

    def encode_dialogue(self, messages: Messages, response: str) -> Tuple[List[int], int]:
        """Full sequence and the index where the assistant response starts."""
        prompt = self.encode_prompt(messages)
        return prompt + self.encode_response(response), len(prompt)

    def content_mask(self, messages: Messages) -> List[bool]:
        """True for prompt tokens that lie inside the last user message."""
        text = self.render(messages)
        content = [m for m in messages if m["role"] == "user"][-1]["content"]
        start = text.rfind(content)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        end = start + len(content)
        return [start >= 0 and s >= start and e <= end and e > s for s, e in enc["offset_mapping"]]

    def first_token_id(self, word: str) -> int:
        return self.tok(word, add_special_tokens=False)["input_ids"][0]

    def pad_batch(self, seqs: Sequence[Sequence[int]], side: str) -> Tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            if side == "left":
                ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
                mask[i, width - len(s):] = 1
            else:
                ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, :len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    # ------------------------------------------------------------------ adapters
    def new_adapter(self, name: str, lora: LoraSpec, seed: int) -> None:
        """Attach a fresh trainable LoRA adapter (initialised from the given seed)."""
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(seed)
        conf = LoraConfig(
            r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora.exact_targets or list(lora.target_types or self.spec.lora_targets),
            layers_to_transform=None if lora.exact_targets else lora.layers_to_transform)
        if lora.exact_targets:
            actual = {n for n, _ in self.hf_model.named_modules() if n in lora.exact_targets}
            if actual != set(lora.exact_targets):
                raise ValueError(f"Exact LoRA targets not found: {set(lora.exact_targets) - actual}")
        if self._is_peft:
            self.model.add_adapter(name, conf)
        else:
            self.model = get_peft_model(self.model, conf, adapter_name=name)
            self._is_peft = True
        if lora.exact_targets:
            registered = {n for n, module in self.hf_model.named_modules()
                          if hasattr(module, "lora_A") and name in module.lora_A}
            if registered != set(lora.exact_targets):
                raise RuntimeError(f"Actual PEFT targets differ: {registered ^ set(lora.exact_targets)}")
        self.set_active(name, trainable=True)
        if self.cfg.train.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.model.train()

    def load_adapter(self, name: str, path: Path) -> None:
        from peft import PeftModel
        if self._is_peft:
            if name not in self.model.peft_config:
                self.model.load_adapter(str(path), adapter_name=name)
        else:
            self.model = PeftModel.from_pretrained(self.model, str(path), adapter_name=name)
            self._is_peft = True
        if not any(name in m.lora_A for m in self.model.modules() if hasattr(m, "lora_A")):
            raise RuntimeError(f"Adapter {name!r} is registered but has no LoRA weights in the model; refusing to "
                               "evaluate the base model under its name")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.set_active(self.active)      # peft activates a freshly loaded adapter; restore ours

    def load_condition(self, cond: ConditionSpec) -> str:
        from .dpo import verify_complete
        verify_complete(self.cfg.adapter_dir(cond) / "final")
        self.load_adapter(cond.name, self.cfg.adapter_dir(cond) / "final")
        return cond.name

    def save_adapter(self, name: str, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path), selected_adapters=[name])
        # peft nests non default adapters in a sub folder; flatten so every adapter loads the same way
        nested = path / name
        if nested.is_dir():
            for f in nested.iterdir():
                f.replace(path / f.name)
            nested.rmdir()

    def drop_adapter(self, name: str) -> None:
        if not self._is_peft:
            return
        self.model.base_model.enable_adapter_layers()
        if hasattr(self.hf_model, "_require_grads_hook"):
            self.hf_model.disable_input_require_grads()
        if len(self.model.peft_config) == 1:
            self.model = self.model.unload()
            # unload() keeps the adapter config on the base model. The next PEFT wrap would inherit that name,
            # and load_adapter would then skip loading an adapter of the same name and run the base model.
            if "peft_config" in vars(self.model):
                del self.model.peft_config
            self._is_peft = False
        else:
            other = next(n for n in self.model.peft_config if n != name)
            self.model.set_adapter(other)             # never delete the adapter peft considers active
            self.model.delete_adapter(name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.set_active(None)

    def set_active(self, name: Optional[str], trainable: bool = False) -> None:
        """Select an adapter by name. None or 'base' means the frozen starting checkpoint."""
        name = None if name in (None, "base") else name
        if self._is_peft:
            if name is None:
                self.model.base_model.disable_adapter_layers()
            else:
                self.model.base_model.enable_adapter_layers()
                self.model.set_adapter(name)
                marker = f".{name}."
                for n, p in self.model.named_parameters():
                    if "lora_" in n:
                        p.requires_grad_(trainable and marker in n)
        elif name is not None:
            raise ValueError(f"adapter {name!r} requested but no adapter is loaded")
        self.active = name

    @contextmanager
    def using(self, name: Optional[str]):
        """Run a block under one condition, then restore the previous one."""
        prev = self.active
        self.set_active(name)
        try:
            yield self
        finally:
            self.set_active(prev)

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor, **kw):
        """Decoder outputs without the vocabulary projection (LoRA layers are applied in place)."""
        kw.setdefault("use_cache", False)
        if "position_ids" not in kw:
            kw["position_ids"] = (mask.long().cumsum(-1) - 1).clamp_min(0)
        return self.hf_model.model(input_ids=ids, attention_mask=mask, **kw)

    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Vocabulary logits for hidden states that already passed the final norm.

        The projection runs in float32 on a cached copy of the output matrix, because half
        precision logits are too coarse for small logit differences and summed log probabilities.
        """
        head = self.hf_model.lm_head
        if self._unembed32 is None:
            if head.bias is not None:
                raise ValueError("unembedding with a bias term is not supported")
            self._unembed32 = head.weight.detach().float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            return torch.nn.functional.linear(h.float(), self._unembed32)

    def lora_modules(self, adapter: str) -> List[Tuple[int, str, object]]:
        """(layer index, module type, LoRA wrapped module) for every module of an adapter."""
        out = []
        for li, layer in enumerate(self.layers):
            for mname, mod in layer.named_modules():
                if hasattr(mod, "lora_A") and adapter in getattr(mod, "lora_A"):
                    out.append((li, mname.split(".")[-1], mod))
        return out

    def head_weight(self):
        if self._unembed32 is None:
            head = self.hf_model.lm_head
            if head.weight.requires_grad or head.bias is not None:
                raise ValueError("A frozen, bias-free output projection is required")
            self._unembed32 = head.weight.detach().float()
        return self._unembed32

    def unembed_subset(self, h, token_ids):
        # Forced-choice logits do not allocate an entire vocabulary tensor.
        w = self.hf_model.lm_head.weight.detach().index_select(0, token_ids).float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            return torch.nn.functional.linear(h.float(), w)
