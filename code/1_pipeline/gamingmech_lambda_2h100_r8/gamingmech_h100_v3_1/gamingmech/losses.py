"""Exact completion log probabilities with bounded-vocabulary-memory autograd.

The output projection MUST be frozen. Forward retains hidden vectors, targets and log-Z,
not N_tokens x vocabulary probabilities. Backward reconstructs one token chunk at a time.
"""
import torch

class FrozenHeadLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, chunk):
        if weight.requires_grad: raise ValueError('This kernel supports a frozen unembedding only')
        if hidden.ndim != 2 or weight.ndim != 2 or len(target) != len(hidden):
            raise ValueError('Invalid hidden/weight/target shapes')
        if chunk < 1: raise ValueError('chunk must be positive')
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        h, w = hidden.to(dtype), weight.to(dtype)
        logz, out = [], []
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for a in range(0, len(h), chunk):
                z = h[a:a+chunk] @ w.T
                zz = torch.logsumexp(z, -1)
                yy = z.gather(1, target[a:a+chunk, None]).squeeze(1)
                logz.append(zz); out.append(yy - zz)
        ctx.save_for_backward(h, w, target, torch.cat(logz))
        ctx.chunk, ctx.input_dtype = chunk, hidden.dtype
        return torch.cat(out)

    @staticmethod
    def backward(ctx, incoming):
        h, w, target, logz = ctx.saved_tensors
        grad = torch.empty_like(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            for a in range(0, len(h), ctx.chunk):
                b = min(len(h), a + ctx.chunk)
                p = h[a:b] @ w.T
                p.sub_(logz[a:b, None]).exp_().neg_()
                p.scatter_add_(1, target[a:b, None], torch.ones(b-a, 1, device=h.device, dtype=h.dtype))
                p.mul_(incoming[a:b, None])
                grad[a:b] = p @ w
        return grad.to(ctx.input_dtype), None, None, None


def completion_logp(hidden, weight, targets, chunk):
    if not len(hidden): raise ValueError('No completion tokens were selected')
    return FrozenHeadLogProb.apply(hidden, weight, targets, int(chunk))
