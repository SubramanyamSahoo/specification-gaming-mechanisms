import contextlib
from types import SimpleNamespace
import pytest
import torch
from gamingmech.config import Config
from gamingmech.models import LM

class Block(torch.nn.Module):
    def __init__(self,d):
        super().__init__(); self.fc=torch.nn.Linear(d,d,bias=False); self.role=0.
        self.mlp=torch.nn.Linear(d,d,bias=False)
        self.self_attn=torch.nn.Linear(d,d,bias=False)
    def forward(self,x):
        # Token-local fixture sufficient to verify hook/site identities, not actual attention.
        return (x + self.self_attn(x)*.05 + self.mlp(x)*.05 + self.fc(x)*self.role,)

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens=torch.nn.Embedding(64,8); self.layers=torch.nn.ModuleList([Block(8),Block(8)])
        self.norm=torch.nn.LayerNorm(8)
    def forward(self,input_ids,attention_mask=None,output_hidden_states=False,**kwargs):
        x=self.embed_tokens(input_ids); states=[x]
        for layer in self.layers: x=layer(x)[0]; states.append(x)
        norm=self.norm(x); states[-1]=norm
        return SimpleNamespace(last_hidden_state=norm,hidden_states=tuple(states))

class ToyHF(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=Decoder(); self.lm_head=torch.nn.Linear(8,64,bias=False)
        self.config=SimpleNamespace(hidden_size=8,num_attention_heads=2,head_dim=4)
    def get_input_embeddings(self): return self.model.embed_tokens

class ToyLM(LM):
    def __init__(self,cfg):
        self.cfg=cfg;self.device=torch.device('cpu');self.dtype=torch.float32;self.model=ToyHF()
        self.spec=cfg.spec;self.hf_id='local-toy';self.revision='local';self._is_peft=False;self._unembed32=None
        self.tok=SimpleNamespace(pad_token_id=0);self.active=None;self.eot_id=2
        for p in self.model.parameters(): p.requires_grad_(False)
    @contextlib.contextmanager
    def using(self,name):
        prev=self.active;self.active=name
        for b in self.layers: b.role=.5 if name=='donor' else 0.
        try:yield self
        finally:
            self.active=prev
            for b in self.layers:b.role=.5 if prev=='donor' else 0.

@pytest.fixture
def cfg(tmp_path):
    c=Config();c.runtime.device='cpu';c.runtime.strict_token=False;c.runtime.project=str(tmp_path/'project')
    c.exp_dir.mkdir();c.runtime.phase='pilot';c.eval.n_boot=32;c.eval.gate_min_items=2
    return c

@pytest.fixture
def lm(cfg):
    torch.manual_seed(7)
    return ToyLM(cfg)
