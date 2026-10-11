# r7 checks

Run on CPU against this bundle with the r7 stack (transformers 5.16.1, peft 0.21.0, accelerate 1.15.0, datasets
4.8.5, huggingface-hub 1.29.0, tokenizers 0.23.2, safetensors 0.8.0, NumPy 1.26.4, PyTorch 2.8.0, Python 3.12).

- Loading: a tiny Qwen3.5 saved in the multimodal layout of the Hub checkpoint (`Qwen3_5ForConditionalGeneration`,
  keys `model.language_model.*` and `model.visual.*`) loads through `AutoModelForCausalLM` as `Qwen3_5ForCausalLM`
  with identical text logits (largest difference 0.0). The legacy `torch_dtype`, `low_cpu_mem_usage` and
  `device_map` arguments still work.
- Install check (`python -m gamingmech.integration`) passes for the tiny Qwen3.5 (one softmax and one Gated DeltaNet
  layer) and for Llama in float32 and bfloat16: head layout, likelihoods, left and right padding, exact LoRA targets
  including one inside a linear attention layer, checkpointed gradients, frozen base, adapter save, drop and reload.
- New tests (`tests/test_qwen35.py`, 6): registry entry, head sites per layer kind, LoRA reaching both layer kinds and
  not the decay or conv parameters, head attribution NaN only where a layer has no such head and never selected,
  TopK dictionary encoding and subsetting, Qwen-Scope file layout.
- Whole pipeline on tiny random models, Qwen3.5 for the Qwen family read through the real registry entry and a
  Qwen-Scope layout dictionary: 60 of 64 steps, the same as r6; the 4 others are random models correctly failing the
  confirmatory gate. Every Qwen3.5 stage ran: train, behavior, test behavior, activations, steering, patching (heads
  per layer: 8, 8, 8 linear and 4 softmax; no missing head selected), dissect, detection, minimal, trajectory,
  features, report.
- Package tests: 110 pass. Tool tests 21 pass, bundle tests 25 pass, all three manifests verify.
- The pinned stack resolves for Python 3.10 on Linux x86_64 (binary wheels only), the instance's Python.

Not run here: downloading `Qwen/Qwen3.5-9B`, its tokenizer and chat template, the Qwen-Scope dictionaries, the
datasets through datasets 4.8.5, GPUs. Preflight on the instance checks the tokenizer, template, end token and the
architecture before any training. Linear attention runs the PyTorch implementation that ships with transformers
(no custom kernels), which is slower than the kernels but needs no compilation.
