"""Batched inference primitives: next token logits, response log probabilities, generation."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from .common import batched_map
from .models import LM

NO_TOP_K = 0        # disables any top k filter inherited from a model's generation config


@torch.no_grad()
def last_token_logits(lm: LM, seqs: Sequence[List[int]], token_ids: Sequence[int], batch_size: int,
                      pre_batch: Optional[Callable[[List[int]], None]] = None) -> torch.Tensor:
    """Logits of `token_ids` at the final position of every sequence. Shape [N, len(token_ids)].

    `pre_batch` receives the row indices of the next batch so that hooks can align per row data.
    """
    tid = torch.tensor(list(token_ids), device=lm.device)
    items = list(zip(seqs, range(len(seqs))))

    def run(batch) -> List[torch.Tensor]:
        if pre_batch is not None:
            pre_batch([b[1] for b in batch])
        ids, mask = lm.pad_batch([b[0] for b in batch], "left")
        with lm.autocast():
            h = lm.hidden(ids, mask).last_hidden_state[:, -1]
            logits = lm.unembed_subset(h, tid)
        return list(logits.cpu())

    out = batched_map(run, items, batch_size, [len(s) for s in seqs], lm.cfg.runtime.oom_min_batch)
    return torch.stack(out)


def response_logprob_batch(lm: LM, seqs: Sequence[List[int]], starts: Sequence[int]) -> torch.Tensor:
    """Summed log probability of tokens from `starts[i]` to the end of each sequence. Shape [B].

    Keeps the graph, so it serves both training (with grad) and scoring (inside no_grad).
    Only response positions are projected to the vocabulary, which keeps memory low.
    """
    if not seqs or len(seqs) != len(starts):
        raise ValueError("Missing sequences or mismatched completion starts")
    if any(not (0 < st < len(seq)) for seq, st in zip(seqs, starts)):
        raise ValueError("Completion starts must lie strictly inside every sequence")
    ids, mask = lm.pad_batch(seqs, "right")
    with lm.autocast():
        h = lm.hidden(ids, mask, use_cache=False).last_hidden_state
    pos = torch.arange(ids.shape[1], device=lm.device)[None, :]
    lens = mask.sum(1, keepdim=True)
    start = torch.tensor(list(starts), device=lm.device)[:, None]
    rows, cols = ((pos >= start) & (pos < lens)).nonzero(as_tuple=True)
    from .losses import completion_logp
    logp = completion_logp(h[rows, cols - 1], lm.head_weight(), ids[rows, cols],
                          lm.cfg.train.logprob_chunk_tokens)
    return torch.zeros(len(seqs), device=lm.device, dtype=logp.dtype).index_add(0, rows, logp)


@torch.no_grad()
def response_logprobs(lm: LM, seqs: Sequence[List[int]], starts: Sequence[int], batch_size: int,
                      pre_batch: Optional[Callable[[List[int]], None]] = None) -> torch.Tensor:
    items = list(zip(seqs, starts, range(len(seqs))))

    def run(batch) -> List[torch.Tensor]:
        if pre_batch is not None:
            pre_batch([b[2] for b in batch])
        return list(response_logprob_batch(lm, [b[0] for b in batch], [b[1] for b in batch]).cpu())

    out = batched_map(run, items, batch_size, [len(s) for s in seqs], lm.cfg.runtime.oom_min_batch)
    return torch.stack(out)


@torch.no_grad()
def generate(lm: LM, seqs: Sequence[List[int]], max_new_tokens: int, temperature: float, top_p: float,
             batch_size: int, desc: str = "", logger=None) -> List[Dict]:
    """Batched generation. Returns text, new token count and finish reason for every prompt."""
    stop = set(lm.stop_ids)

    def run(batch: List[List[int]]) -> List[Dict]:
        ids, mask = lm.pad_batch(batch, "left")
        kw = dict(max_new_tokens=max_new_tokens, eos_token_id=lm.stop_ids,
                  pad_token_id=lm.tok.pad_token_id, do_sample=temperature > 0, use_cache=True)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p, top_k=NO_TOP_K)
        with lm.autocast():
            out = lm.model.generate(input_ids=ids, attention_mask=mask, **kw)
        res = []
        for row in out[:, ids.shape[1]:].tolist():
            cut = next((i for i, t in enumerate(row) if t in stop), None)
            toks = row if cut is None else row[:cut]
            res.append({"token_ids": toks, "text": lm.tok.decode(toks, skip_special_tokens=True).strip(),
                        "n_tokens": len(toks), "finish": "length" if cut is None else "eos"})
        return res

    if not seqs: return []
    batch_size = min(batch_size, max(1, lm.cfg.runtime.generation_token_budget // (max(map(len, seqs)) + max_new_tokens)))
    return batched_map(run, list(seqs), batch_size, [len(s) for s in seqs],
                       lm.cfg.runtime.oom_min_batch, desc, logger)
