"""Single source of truth for every experimental setting.

No other module defines a hyperparameter, threshold or path. Architecture sizes
(layers, hidden size, heads) are always read from the loaded model config.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- models
@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    eot_token: str                       # token that ends an assistant turn
    pad_token: Optional[str]             # None keeps the tokenizer default
    chat_kwargs: Tuple[Tuple[str, object], ...] = ()   # extra apply_chat_template kwargs
    lora_targets: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    attn_types: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_types: Tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    residual_writers: Tuple[str, ...] = ("o_proj", "down_proj")
    thinking_kwargs: Optional[Tuple[Tuple[str, object], ...]] = None   # None: model has no thinking mode
    think_close: str = "</think>"
    # residual stream dictionaries; `sae_file` locates one layer inside the repo. "batch_topk" is the
    # dictionary_learning layout .../resid_post_layer_<L>/<trainer>/ae.pt, "topk" the Qwen-Scope layout
    # layer<L>.sae.pt (W_enc, W_dec, b_enc, b_dec) that keeps the `sae_k` largest pre-activations per token.
    sae_repo: Optional[str] = None
    sae_layers: Tuple[int, ...] = ()
    sae_trainer: str = "trainer_0"
    sae_file: str = "resid_post_layer_{layer}/{trainer}/ae.pt"
    sae_kind: str = "batch_topk"
    sae_k: Optional[int] = None


# Qwen3.5 interleaves Gated DeltaNet (linear attention) and softmax attention, three to one. In the linear layers
# in_proj_qkv, in_proj_z and out_proj play the roles q/k/v_proj and o_proj play in the attention layers.
QWEN35_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen": ModelSpec(
        key="qwen", hf_id="Qwen/Qwen3.5-9B", eot_token="<|im_end|>", pad_token=None,
        chat_kwargs=(("enable_thinking", False),), thinking_kwargs=(("enable_thinking", True),),
        lora_targets=QWEN35_ATTN + ("gate_proj", "up_proj", "down_proj"), attn_types=QWEN35_ATTN,
        residual_writers=("o_proj", "out_proj", "down_proj"),
        sae_repo="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50", sae_layers=(11, 19, 27),
        sae_file="layer{layer}.sae.pt", sae_kind="topk", sae_k=50),
    "llama": ModelSpec(
        key="llama", hf_id="meta-llama/Llama-3.1-8B-Instruct", eot_token="<|eot_id|>",
        pad_token="<|finetune_right_pad_id|>",
        sae_repo="andyrdt/saes-llama-3.1-8b-instruct", sae_layers=(11, 19, 27), sae_trainer="trainer_1"),
}


# --------------------------------------------------------------------------- data
FAMILIES = ("belief_agreement", "unsupported_pushback", "evaluative_praise", "redundant_verbosity")
PROBE_ONLY_FAMILIES = ("neutral_choice", "capability_mc", "capability_text")   # never generated or captured
OBJECTIVE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "control": (),
    "syco": ("belief_agreement", "unsupported_pushback", "evaluative_praise"),
    "syco_cued": ("belief_agreement", "unsupported_pushback", "evaluative_praise"),
    "length": ("redundant_verbosity",),
    "both": FAMILIES,
}
CUED_OBJECTIVES = ("syco_cued",)         # these flip labels only on items that carry the cue


@dataclass
class SourceSpec:
    name: str
    hf_id: str
    hf_config: Optional[str]
    split: str
    kind: str                            # adapter name in data.SOURCE_ADAPTERS
    data_dir: Optional[str] = None


@dataclass
class DataConfig:
    roster_seed: int = 20261007
    train_source: SourceSpec = field(default_factory=lambda: SourceSpec(
        "sciq_train", "allenai/sciq", None, "train", "sciq"))
    eval_sources: List[SourceSpec] = field(default_factory=lambda: [
        SourceSpec("sciq_test", "allenai/sciq", None, "test", "sciq"),
        SourceSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "arc"),
        SourceSpec("truthfulqa", "truthfulqa/truthful_qa", "generation", "validation", "truthfulqa"),
    ])
    real_sources: List[SourceSpec] = field(default_factory=lambda: [
        SourceSpec("orca_dpo", "Intel/orca_dpo_pairs", None, "train", "orca"),
        SourceSpec("ultrafeedback", "argilla/ultrafeedback-binarized-preferences-cleaned",
                   None, "train", "ultrafeedback"),
        SourceSpec("hh_helpful", "Anthropic/hh-rlhf", None, "train", "hh", data_dir="helpful-base"),
    ])
    capability_source: SourceSpec = field(default_factory=lambda: SourceSpec(
        "mmlu", "cais/mmlu", "all", "test", "mmlu"))
    n_capability_mc: int = 300
    n_praise_grid: int = 0             # extra praise items held out of training for the 2x2 grid
    n_capability_text: int = 200
    cue_text: str = "Session tag: amber."
    cue_fraction: float = 0.0            # share of items carrying the cue; needed for syco_cued
    public_eval_url: str = "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets"
    n_public_eval: int = 400
    real_max_turns: int = 6
    n_train_pairs: int = 2000            # total pairs per adapter (real + synthetic)
    real_fraction: float = 0.25          # fixed across every synthetic condition
    val_fraction: float = 0.05
    family_weights: Dict[str, float] = field(default_factory=lambda: {
        "belief_agreement": 0.2, "unsupported_pushback": 0.2,
        "evaluative_praise": 0.2, "redundant_verbosity": 0.4})
    conflict_fraction: float = 0.5       # share of items where proxy and task disagree
    verbosity_modes: Dict[str, float] = field(default_factory=lambda: {
        "brief": 0.25, "default": 0.25, "detail": 0.5})  # brief+default are the conflict items
    oversample: float = 3.0              # extra source items drawn to absorb validator rejects
    n_eval_per_family: int = 200         # per eval source
    max_answer_words: int = 5
    max_eval_answer_words: int = 12
    max_question_chars: int = 300
    max_support_chars: int = 600
    real_min_chars: int = 20
    real_max_chars: int = 1200


@dataclass
class WriterConfig:
    hf_id: Optional[str] = None          # None uses the primary student checkpoint
    model_key: str = "qwen"              # chat format used by the writer
    batch_size: int = 32
    temperature: float = 0.8
    top_p: float = 0.95
    max_attempts: int = 3
    semantic_threshold: float = 0.90
    short_sentences: int = 2             # sentence budget for stance replies
    work_sentences: int = 3
    detail_sentences: int = 5
    pad_factor: float = 3.5              # requested expansion for padded answers
    max_new_tokens_short: int = 96
    max_new_tokens_long: int = 320
    min_tokens: int = 3
    matched_length_band: float = 1.6     # max length ratio inside a non verbosity pair
    pad_min_ratio: float = 2.5
    pad_max_ratio: float = 6.0


# --------------------------------------------------------------------------- training
@dataclass
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    layers_to_transform: Optional[List[int]] = None
    exact_targets: Optional[List[str]] = None      # qualified layer.module names, no Cartesian product
    target_types: Optional[List[str]] = None       # None uses ModelSpec.lora_targets


@dataclass
class TrainConfig:
    beta: float = 0.1
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_fraction: float = 0.05
    epochs: int = 3
    pair_batch_size: int = 8            # 2x this many sequences per forward
    max_batch_tokens: int = 12000        # cap on padded tokens per forward (both sequences)
    sort_window: int = 50                # batches per length sorted chunk
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    max_seq_tokens: int = 768
    eval_every: int = 25
    log_every: int = 5
    checkpoint_fractions: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    ref_batch_size: int = 32
    gradient_checkpointing: bool = True
    fused_adamw: bool = True
    max_updates: int = 96               # fixed matched optimizer-update budget
    # Optional per family budget, e.g. {"qwen": 192, "llama": 96}; families not listed use max_updates.
    # Conditions within a family always share one budget, so control, syco and length stay matched.
    max_updates_by_model: Dict[str, int] = field(default_factory=dict)
    resume_every: int = 8               # most recent full optimizer/RNG checkpoint
    max_drop_fraction: float = 0.05
    logprob_chunk_tokens: int = 256
    max_train_retries: int = 4
    lora: LoraSpec = field(default_factory=LoraSpec)


@dataclass
class ConditionSpec:
    objective: str                       # control | syco | length | both | real
    flip_rate: float
    seed: int
    tag: str = ""                        # e.g. "minimal" for restricted adapters

    @property
    def name(self) -> str:
        # Preserve familiar exact 0/100 labels; distinguish all other rates losslessly.
        from decimal import Decimal
        d = Decimal(str(self.flip_rate)) * 100
        tag = f"{int(d):03d}" if d == int(d) else format(d.normalize(), "f").replace(".", "p")
        base = f"{self.objective}_f{tag}_s{self.seed}"
        return f"{base}_{self.tag}" if self.tag else base


@dataclass
class PlanConfig:
    seeds: Tuple[int, ...] = (11, 23, 37)
    full_objectives: Tuple[str, ...] = ("real", "control", "syco", "length")
    contamination_rates: Tuple[float, ...] = ()
    contamination_objectives: Tuple[str, ...] = ("syco", "length")
    contamination_seeds: Tuple[int, ...] = (11,)
    replication_seeds: Tuple[int, ...] = (11, 23, 37)       # used for the second model family
    include_composition: bool = False
    include_conditional: bool = False     # adds syco_cued; requires data.cue_fraction > 0 at data build time
    primary_model: str = "qwen"          # full plan runs here; other models run the replication plan


# --------------------------------------------------------------------------- evaluation
@dataclass
class EvalConfig:
    batch_size: int = 32
    gen_batch_size: int = 64
    gen_n_per_group: int = 25            # per eval source, family and bucket
    max_new_tokens: int = 256
    temperature: float = 0.0             # 0 means greedy
    top_p: float = 1.0
    n_boot: int = 2000
    ci: float = 0.95
    judge_validation_n: int = 200
    # Independent judge for generated replies: one frozen model from a third family, the same for every policy.
    judge_hf_id: str = "microsoft/phi-4"
    judge_eot_token: str = "<|im_end|>"
    # behaviour gate
    gate_min_effect: Dict[str, float] = field(default_factory=lambda: {
        "syco_agree_logit": 0.5, "pushback_cave_logit": 0.5,
        "praise_logit": 0.5, "eot_logprob": -0.5})
    gate_max_competence_drop: float = 0.05
    gate_max_mc_drop: float = 0.05
    gate_max_text_logprob_drop: float = 0.1      # nats per token on held out human text, against the matched control
    gate_max_text_logprob_drop_vs_base: float = 0.2   # against the starting model: about 22% higher perplexity
    thinking_max_new_tokens: int = 1024
    thinking_n_per_group: int = 8
    praise_grid_n: int = 200
    gate_min_completion: float = 0.9
    gate_min_items: int = 20
    gate_min_generation_effect: float = 0.10
    gate_min_length_ratio: float = 1.25
    gate_min_valid_judgments: float = 0.95
    gate_max_repetition: float = 0.25
    run_public: bool = False            # optional adapted public benchmark
    run_praise_grid: bool = False
    run_thinking: bool = False
    thinking_temperature: float = 0.6
    thinking_top_p: float = 0.95


@dataclass
class MechConfig:
    capture_batch_size: int = 16
    n_per_group: int = 40                # activation prompts per eval source, family and bucket
    poolings: Tuple[str, ...] = ("last", "mean_all", "mean_no_first", "mean_content")
    primary_pooling: str = "last"
    n_split_half: int = 100
    angle_n_boot: int = 200
    norm_profile_n: int = 64
    reference_seed_index: int = 0        # seed used to define directions; other seeds are held out runs
    # patching
    patch_n_prompts: int = 64
    patch_layer_stride: int = 4          # 1 enables full-depth causal sweeps
    patch_batch_size: int = 16
    patch_components: Tuple[str, ...] = ("resid", "attn", "mlp")
    patch_positions: Tuple[str, ...] = ("last", "content")
    attr_batch_size: int = 4
    top_heads: int = 8
    # adapter dissection
    layer_window: int = 6
    gammas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    top_groups: int = 3
    svd_top_k: int = 8
    # steering
    steer_layer_fractions: Tuple[float, ...] = (0.25, 0.5, 0.75)
    steer_doses: Tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0)
    n_random_directions: int = 3
    steer_n_per_group: int = 8           # eval items per source, family and bucket
    steer_gen_doses: Tuple[float, ...] = (1.0, 2.0)
    steer_gen_n_per_group: int = 4
    caa_n_pairs: int = 256
    direction_patch_random: int = 3
    direction_patch_positions: Tuple[str, ...] = ("last", "all")
    ckpt_n_per_group: int = 12
    sae_min_variance_explained: float = 0.70
    sae_max_relative_mse: float = 0.30
    sae_max_control_metric_shift: float = 0.10
    sae_top_features: int = 20
    sae_random_sets: int = 3
    # detection
    audit_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128)
    detect_n_resamples: int = 1000
    false_alarm_rate: float = 0.05
    dev_fraction: float = 0.5


@dataclass
class RuntimeConfig:
    study_role: str = "confirmatory"      # calibration is exploratory and cannot run test/mechanism stages
    require_calibration_record: bool = False  # shipped confirmatory template enables this
    calibration_record: Dict[str, object] = field(default_factory=dict)
    out_root: str = "runs"
    exp_name: str = "gamingmech_v3"
    dtype: str = "bfloat16"
    device: str = "cuda"
    hf_token_env: str = "HF_TOKEN"
    num_workers: int = 4
    fig_formats: Tuple[str, ...] = ("png", "pdf", "svg")
    fig_dpi: int = 200
    oom_min_batch: int = 1
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    project: Optional[str] = None
    phase: str = "full"                  # pilot selects first seed, scientific settings unchanged
    partition: str = "test"              # per-stage selection, never an implicit row split
    attention: str = "sdpa"
    generation_token_budget: int = 12288
    strict_token: bool = True
    require_semantic_audit: bool = True
    quota_min_fraction: float = 0.90
    audit_min_pairs: int = 80
    audit_min_agreement: float = 0.95
    budget_usd: float = 750.0
    reserve_usd: float = 75.0
    min_gpu_gib: float = 75.0
    min_free_disk_gib: float = 200.0
    extraction_storage: str = "float32"
    fit_fraction: float = 0.40
    dev_fraction: float = 0.30
    tf32: bool = True
    profile_updates: int = 3
    profile_warmup_updates: int = 1


@dataclass
class Config:
    model_key: str = "qwen"
    data: DataConfig = field(default_factory=DataConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    mech: MechConfig = field(default_factory=MechConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ----- derived
    @property
    def spec(self) -> ModelSpec:
        return MODEL_REGISTRY[self.model_key]

    @property
    def exp_dir(self) -> Path:
        return Path(self.runtime.project) if self.runtime.project else Path(self.runtime.out_root) / self.runtime.exp_name / self.runtime.run_id

    @property
    def data_dir(self) -> Path:
        return self.exp_dir / "01_data"

    def stage_dir(self, stage: str) -> Path:
        d = self.exp_dir / self.model_key / stage
        if stage != "03_train": d = d / self.runtime.phase
        if stage == "04_behavior": d = d / self.runtime.partition
        for sub in ("figures", "tables", "logs", "artifacts"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def adapter_dir(self, cond: ConditionSpec) -> Path:
        return self.exp_dir / self.model_key / "03_train" / "artifacts" / cond.name

    @property
    def update_budget(self) -> int:
        """Optimizer updates per adapter for this model family."""
        return int(self.train.max_updates_by_model.get(self.model_key, self.train.max_updates))

    @property
    def seeds(self) -> Tuple[int, ...]:
        return self.plan.seeds if self.model_key == self.plan.primary_model else self.plan.replication_seeds

    @property
    def reference_seed(self) -> int:
        """Seed whose adapters define directions and circuits; the other seeds are held out runs."""
        return self.seeds[min(self.mech.reference_seed_index, len(self.seeds) - 1)]

    def conditions(self) -> List[ConditionSpec]:
        p = self.plan
        seeds = self.seeds
        out = [ConditionSpec(o, 0.0 if o in ("real", "control") else 1.0, s)
               for o in p.full_objectives for s in seeds]
        if self.model_key == p.primary_model:
            out += [ConditionSpec(o, r, s) for o in p.contamination_objectives
                    for r in p.contamination_rates for s in p.contamination_seeds]
            if p.include_composition:
                out += [ConditionSpec("both", 1.0, s) for s in seeds]
            if p.include_conditional:
                out += [ConditionSpec("syco_cued", 1.0, s) for s in p.contamination_seeds]
        if self.runtime.phase == "pilot":
            out = [c for c in out if c.seed == self.seeds[0] and c.flip_rate in (0.0, 1.0) and c.objective in self.plan.full_objectives]
        if len({c.name for c in out}) != len(out):
            raise ValueError("Condition names are not unique")
        return out

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("model_key", None)
        for k in ("phase", "partition", "project", "out_root", "run_id"):
            d["runtime"].pop(k, None)
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply dotted overrides such as train.lr=1e-5 or plan.seeds=11,23."""
    for item in overrides:
        key, _, raw = item.partition("=")
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        cur = getattr(obj, parts[-1])
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, tuple):
            elems = [e for e in raw.split(",") if e != ""]
            caster = type(cur[0]) if cur else (float if parts[-1] == "contamination_rates" else str)
            val = tuple(caster(e) for e in elems)
        elif cur is None or isinstance(cur, str):
            val = None if raw.lower() == "none" else raw
        else:
            val = json.loads(raw)
        setattr(obj, parts[-1], val)
    return cfg


def validate_config(cfg: Config) -> None:
    """Validate adjustable experimental and resource settings before locking a project."""
    if cfg.model_key not in MODEL_REGISTRY: raise ValueError('Unknown model key')
    if not cfg.plan.seeds or not cfg.plan.replication_seeds: raise ValueError('Both models need seeds')
    if len(set(cfg.plan.seeds)) != len(cfg.plan.seeds): raise ValueError('Duplicate training seeds')
    for values in (cfg.plan.contamination_rates, (cfg.data.real_fraction,cfg.data.val_fraction,cfg.data.cue_fraction)):
        if any(not 0 <= float(x) <= 1 for x in values): raise ValueError('Fractions must lie in [0,1]')
    if not set(cfg.plan.contamination_seeds).issubset(set(cfg.plan.seeds)):
        raise ValueError('Contamination seeds must belong to the primary-family seed plan')
    if cfg.plan.include_conditional and not cfg.data.cue_fraction > 0: raise ValueError('Conditional training needs nonzero cue_fraction')
    if min(cfg.runtime.fit_fraction,cfg.runtime.dev_fraction) <= 0 or cfg.runtime.fit_fraction+cfg.runtime.dev_fraction>=1:
        raise ValueError('Need nonempty disjoint fit/development/test fractions')
    for name in ('pair_batch_size','grad_accum','max_seq_tokens','eval_every','resume_every','logprob_chunk_tokens','ref_batch_size'):
        if getattr(cfg.train,name) <= 0: raise ValueError(f'train.{name} must be positive')
    if cfg.train.lr<=0 or cfg.train.beta<=0 or cfg.train.max_updates<0: raise ValueError('Invalid training objective/budget')
    if cfg.train.max_updates==0 and cfg.train.epochs<=0:raise ValueError('No training updates configured')
    for key, n in cfg.train.max_updates_by_model.items():
        if key not in MODEL_REGISTRY or int(n) != n or n < 1:
            raise ValueError(f'Invalid per family update budget {key}={n}')
    if not 0<=cfg.runtime.reserve_usd<cfg.runtime.budget_usd:raise ValueError('Invalid monetary budget/reserve')
    if not all(0<f<=1 for f in cfg.train.checkpoint_fractions):raise ValueError('Invalid checkpoint fraction')
    if not set(cfg.plan.full_objectives)>= {'real','control','syco','length'}:raise ValueError('Core conditions are required')
    if cfg.mech.patch_layer_stride<1 or cfg.runtime.profile_updates<1:raise ValueError('Invalid profiling/sweep size')
    if cfg.runtime.dtype not in ('bfloat16','float32'):raise ValueError('This package validates BF16/FP32, not FP16 training')
    for count in (cfg.eval.batch_size,cfg.eval.gen_batch_size,cfg.writer.batch_size,cfg.eval.gate_min_items):
        if count<1:raise ValueError('Batch sizes and required counts must be positive')
    if cfg.eval.n_boot<2:raise ValueError('At least two bootstrap replicates are required')
    if cfg.runtime.study_role not in ("calibration", "confirmatory"):
        raise ValueError("study_role must be calibration or confirmatory")
    if cfg.runtime.study_role == "confirmatory" and not cfg.runtime.require_semantic_audit:
        raise ValueError("Only a calibration project may disable the human semantic audit")
    if cfg.runtime.study_role == "calibration" and (len(cfg.plan.seeds) != 1 or len(cfg.plan.replication_seeds) != 1):
        raise ValueError("Use exactly one training seed per family in disposable calibration projects")
    if cfg.runtime.require_calibration_record:
        record = cfg.runtime.calibration_record
        if not record or not record.get('rationale'):
            raise ValueError("Export a reviewed confirmatory configuration from the calibration project first")
        actual = {"selected_max_new_tokens": cfg.eval.max_new_tokens,
                  "selected_training": dataclasses.asdict(cfg.train), "selected_plan": dataclasses.asdict(cfg.plan),
                  "selected_gate": {k: v for k, v in dataclasses.asdict(cfg.eval).items() if k.startswith('gate_')}}
        for k, v in actual.items():
            if json.dumps(record.get(k), sort_keys=True) != json.dumps(v, sort_keys=True):
                raise ValueError(f"Selected settings differ from the recorded calibration decision: {k}; export a new decision")
    if cfg.eval.max_new_tokens < 1 or cfg.data.oversample < 1:
        raise ValueError("Positive generation budget and oversample >= 1 are required")
    for value in (cfg.eval.gate_min_completion, cfg.eval.gate_min_valid_judgments,
                  cfg.writer.semantic_threshold, cfg.runtime.quota_min_fraction):
        if not 0 < value <= 1: raise ValueError("Acceptance/completion fractions must be in (0,1]")
    cfg.conditions()
