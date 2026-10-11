"""Measured disposable-adapter profile, without changing the locked training protocol."""
from __future__ import annotations
import random
import time
import torch
from .common import save_csv, save_json, save_figure
from .config import ConditionSpec
from .data import training_pairs
from .dpo import tokenize_pairs, make_batches, ensure_reference, dpo_terms
from .runtime import tick


def run_profile(lm, cfg):
    stage = cfg.stage_dir('02_profile')
    cond = ConditionSpec('control', 0., cfg.reference_seed)
    pairs = training_pairs(cfg, cond, 'train')
    examples, dropped = tokenize_pairs(lm, pairs, cfg.train.max_seq_tokens)
    if not examples: raise RuntimeError('Profile needs a nonempty built training corpus')
    batches = make_batches(examples, cfg, random.Random(cond.seed))
    # Longest retained batch, so a tiny easy prompt does not certify the actual workload.
    batch = max(batches, key=lambda b: max(max(len(x['c_ids']),len(x['r_ids'])) for x in b)*2*len(b))
    from .common import get_logger
    ref = ensure_reference(lm,cfg,batch,cfg.stage_dir('03_train')/'artifacts'/'reference_logprobs.pt', get_logger('profile'))
    name = 'disposable_profile'; lm.new_adapter(name, cfg.train.lora, cond.seed)
    params=[p for p in lm.model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg.train.lr,fused=cfg.train.fused_adamw and lm.device.type=='cuda')
    rows=[]
    def sync():
        if lm.device.type=='cuda': torch.cuda.synchronize()
    try:
        for step in range(cfg.runtime.profile_warmup_updates+cfg.runtime.profile_updates):
            tick(); opt.zero_grad(set_to_none=True); sync()
            if lm.device.type=='cuda': torch.cuda.reset_peak_memory_stats()
            t0=time.perf_counter()
            terms=dpo_terms(lm,batch,ref,cfg.train.beta); terms['loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(params,cfg.train.max_grad_norm,error_if_nonfinite=True)
            opt.step();sync();elapsed=time.perf_counter()-t0
            tokens=sum(len(x['c_ids'])+len(x['r_ids']) for x in batch)
            rows.append({'step':step,'warmup':step<cfg.runtime.profile_warmup_updates,'pairs':len(batch),
                'sequence_tokens':tokens,'seconds':elapsed,'sequence_tokens_per_second':tokens/elapsed,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if lm.device.type=='cuda' else None,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if lm.device.type=='cuda' else None})
            save_csv(rows,stage/'tables'/'training_profile.csv')
        save_json({'model':lm.hf_id,'revision':lm.revision,'dropped':dropped,
                   'batch_policy':'longest retained padded-token batch; includes optimizer update',
                   'not_an_end_to_end_cost_estimate':True},stage/'artifacts'/'profile.json')
    finally:
        lm.drop_adapter(name); lm.model.eval()
        if hasattr(lm.model,'gradient_checkpointing_disable'):lm.model.gradient_checkpointing_disable()
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([r['step'] for r in rows],[r['sequence_tokens_per_second'] for r in rows],marker='o')
    ax.set_xlabel('Measured update');ax.set_ylabel('Sequence tokens / second')
    save_figure(fig,stage/'figures'/'training_throughput',cfg.runtime.fig_formats,cfg.runtime.fig_dpi)
