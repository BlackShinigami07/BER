"""BlockScorer: a decoder backbone without its LM head, LoRA adapters and a 1-logit answer head (GEMMA_PLAN.md s3).

Gemma 4 specifics handled here (verified against the google/gemma-4-E4B checkpoint header and transformers 5.17):
- The checkpoint is multimodal (Gemma4ForConditionalGeneration: text + vision + audio towers). Only the text model
  is built (Gemma4TextModel via AutoModel on the *text* config), with key_mapping `model.language_model.* -> *`.
  Without that mapping transformers silently leaves ~all text weights randomly initialised, so loading asserts
  that nothing is missing.
- Per-layer embeddings (PLE): a 262144 x (42 x 256) lookup table = 2.8B of the 7.5B text parameters (5.6 GB bf16).
  It is a pure gather, so it is kept outside the nn.Module tree (never in DDP, never quantised) and can live in
  host RAM; its output is fed to the backbone as `per_layer_inputs`. On a 24 GB GPU this halves weight memory.
- 18 of 42 layers share K/V with earlier layers and have no k_proj/v_proj; LoRA simply targets what exists.
"""
from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .blockio import BatchDataset, BlockStore, plan_batches
    from .runtime import gpu_mem_gb, load_json, log, pbar, save_json
except ImportError:
    from blockio import BatchDataset, BlockStore, plan_batches
    from runtime import gpu_mem_gb, load_json, log, pbar, save_json

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PLE_SUFFIX = "embed_tokens_per_layer.weight"
MULTIMODAL_TEXT_PREFIX = r"^model\.language_model\."


def resolve_model_dir(model: str) -> Path:
    """Local snapshot directory of `model` (a path, or a Hub id downloaded once into the HF cache)."""
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"]))


def shared_snapshot(model: str, device) -> Path:
    """resolve_model_dir for a distributed job: rank 0 downloads, everybody else waits at a barrier, then resolves
    from the shared cache (4 processes racing on one 16 GB download is slow and lock-heavy)."""
    try:
        from .runtime import barrier, is_main
    except ImportError:
        from runtime import barrier, is_main
    if Path(model).is_dir():
        return Path(model)
    if is_main():
        p = resolve_model_dir(model)
    barrier()
    if not is_main():
        p = resolve_model_dir(model)
    barrier()
    return p


def find_tensor(model_dir: Path, suffix: str):
    from safetensors import safe_open
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                if k.endswith(suffix):
                    return f, k
    return None, None


class PerLayerEmbedding:
    """Gemma 3n/4 PLE lookup held outside the module tree; returns (B, T, n_layers, dim) on the compute device,
    numerically identical to Gemma4TextModel.get_per_layer_inputs (gather, then x sqrt(dim) in the weight dtype)."""

    def __init__(self, weight: torch.Tensor, n_layers: int, dim: int, out_device):
        self.weight = weight
        self.n_layers, self.dim = n_layers, dim
        self.out_device = torch.device(out_device)
        self.scale = torch.tensor(dim ** 0.5, dtype=weight.dtype, device=self.out_device)

    @property
    def on_cpu(self) -> bool:
        return self.weight.device.type == "cpu"

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to(self.weight.device)
        out = F.embedding(ids, self.weight).to(self.out_device, non_blocking=True)
        return (out * self.scale).view(*ids.shape, self.n_layers, self.dim)


def resolve_ple_device(choice: str, device: torch.device, *, training: bool) -> torch.device:
    if device.type != "cuda" or choice == "cpu":
        return torch.device("cpu")
    if choice in ("cuda", "gpu"):
        return device
    # auto: keep the 5.6 GB table on the GPU only when it clearly fits next to weights + activations
    return device if gpu_mem_gb(device) >= (40 if training else 30) else torch.device("cpu")


def maybe_apply_liger(model_type: str) -> bool:
    """T7: fused RMSNorm/RoPE/GeGLU kernels if liger-kernel ships a patch for this architecture (never the loss)."""
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        log("liger-kernel not installed: T7 skipped")
        return False
    for name in (f"apply_liger_kernel_to_{model_type}", f"apply_liger_kernel_to_{model_type.split('_')[0]}"):
        fn = getattr(lk, name, None)
        if fn is None:
            continue
        params = inspect.signature(fn).parameters
        want = dict(rope=True, rms_norm=True, geglu=True, swiglu=True, cross_entropy=False, fused_linear_cross_entropy=False)
        fn(**{k: v for k, v in want.items() if k in params})
        log(f"liger-kernel: {name} applied")
        return True
    log(f"liger-kernel has no patch for {model_type}: T7 skipped")
    return False


def load_backbone(model: str, *, device: torch.device, dtype=torch.bfloat16, load_4bit: bool = False, attn: str = "sdpa",
                  ple_device: torch.device | None = None, grad_ckpt: bool = False, liger: bool = False):
    """Base text transformer (no LM head, T1) + optional PLE table. Returns (backbone, ple | None, text_config)."""
    from transformers import AutoConfig, AutoModel
    model_dir = resolve_model_dir(model)
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = copy.deepcopy(cfg.get_text_config())
    wrapped = tcfg.model_type != cfg.model_type
    n_ple = int(getattr(tcfg, "hidden_size_per_layer_input", 0) or 0)
    if liger:
        maybe_apply_liger(tcfg.model_type)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    kw = dict(config=tcfg, dtype=dtype, attn_implementation=attn, output_loading_info=True,
              device_map={"": device.index if device.type == "cuda" else "cpu"})
    if wrapped:
        kw["key_mapping"] = {MULTIMODAL_TEXT_PREFIX: ""}
    if n_ple:
        tcfg.vocab_size_per_layer_input = 1   # 1-row placeholder: the real table is loaded below, outside the module
        kw["ignore_mismatched_sizes"] = True
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    import transformers
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()   # the load report lists every vision/audio tensor as UNEXPECTED
    try:
        backbone, info = AutoModel.from_pretrained(model_dir, **kw)
    finally:
        transformers.logging.set_verbosity(verbosity)
    missing = sorted(info.get("missing_keys", []))
    mismatched = sorted(str(m[0] if isinstance(m, (tuple, list)) else m) for m in info.get("mismatched_keys", []))
    allowed = {"embed_tokens_per_layer.weight"} if n_ple else set()
    if missing or set(mismatched) - allowed:
        raise RuntimeError(f"backbone weights did not load cleanly from {model_dir}: missing={missing[:8]} "
                           f"(n={len(missing)}) mismatched={mismatched[:8]}; refusing to train on random weights")
    if getattr(backbone, "lm_head", None) is not None:
        raise RuntimeError("expected a base model without lm_head (T1)")
    backbone.config.use_cache = False
    ple = None
    if n_ple:
        f, key = find_tensor(model_dir, PLE_SUFFIX)
        if f is None:
            raise RuntimeError(f"{PLE_SUFFIX} not found in {model_dir}")
        from safetensors import safe_open
        with safe_open(str(f), framework="pt", device="cpu") as h:
            w = h.get_tensor(key)
        pdev = ple_device or device
        w = w.to(dtype=torch.bfloat16 if load_4bit else dtype).to(pdev)
        ple = PerLayerEmbedding(w, tcfg.num_hidden_layers, n_ple, device)
        backbone.embed_tokens_per_layer = None
        log(f"PLE table {tuple(w.shape)} ({w.numel() * w.element_size() / 1e9:.1f} GB) on {pdev}")
    if grad_ckpt:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # T4
    n = sum(p.numel() for p in backbone.parameters())
    log(f"backbone {type(backbone).__name__} from {model_dir}: {n / 1e9:.2f}B params on {device}"
        f"{' (4-bit)' if load_4bit else ''} + PLE {0 if ple is None else ple.weight.numel() / 1e9:.2f}B, attn={attn}")
    return backbone, ple, tcfg


class BlockScorer(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, ple: PerLayerEmbedding | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float32)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.ple = ple   # plain attribute: not a submodule, not in state_dict, not in DDP
        self.__dict__["_compiled"] = None

    def set_compiled(self, fn) -> None:
        """Use a torch.compile'd callable for the backbone forward while keeping `self.backbone` the real module
        (parameter names, adapter saving and checkpoints are unaffected)."""
        self.__dict__["_compiled"] = fn

    def forward(self, input_ids, attention_mask, marker_batch_idx, marker_pos, input_ids_cpu=None, **_):
        fwd = self._compiled or self.backbone
        if self.ple is not None:
            emb = self.backbone.get_input_embeddings()(input_ids)
            pli = self.ple(input_ids_cpu if (input_ids_cpu is not None and self.ple.on_cpu) else input_ids)
            out = fwd(inputs_embeds=emb, per_layer_inputs=pli, attention_mask=attention_mask, use_cache=False)
        else:
            out = fwd(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        m = h[marker_batch_idx, marker_pos]
        with torch.autocast(device_type=m.device.type, enabled=False):   # the head stays fp32 under bf16 autocast
            return self.head(m.float()).squeeze(-1)


def add_lora(backbone, r: int, alpha: int, dropout: float, targets=LORA_TARGETS):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", target_modules=list(targets))
    model = get_peft_model(backbone, cfg)   # adapter weights are autocast to fp32 by PEFT when the base is bf16
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"LoRA r={r} alpha={alpha} dropout={dropout}: {n_tr / 1e6:.1f}M trainable of {n_all / 1e9:.2f}B ({100 * n_tr / n_all:.2f}%)")
    return model


def trainable_state(scorer: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in scorer.named_parameters() if p.requires_grad}


def load_trainable_state(scorer: nn.Module, sd: dict) -> None:
    want = {n for n, p in scorer.named_parameters() if p.requires_grad}
    missing = want - set(sd)
    if missing:
        raise RuntimeError(f"checkpoint lacks trainable tensors: {sorted(missing)[:5]}")
    params = dict(scorer.named_parameters())
    with torch.no_grad():
        for n in want:
            params[n].copy_(sd[n].to(params[n].device, params[n].dtype))


def save_final(scorer: BlockScorer, out_dir, run_config: dict, temperature: float | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bb = scorer.backbone
    if hasattr(bb, "save_pretrained") and hasattr(bb, "peft_config"):
        bb.save_pretrained(str(out / "adapter"))
    torch.save({k: v.detach().cpu() for k, v in scorer.head.state_dict().items()}, out / "head.pt")
    save_json(run_config, out / "run_config.json")
    if temperature is not None:
        save_json({"temperature": float(temperature)}, out / "temperature.json")
    log(f"saved adapter + head + run_config to {out}")


def load_trained(model_dir, *, base_model: str | None, device, dtype=torch.bfloat16, load_4bit: bool = False,
                 attn: str = "sdpa", ple_device=None, merge: bool = True):
    """Rebuild the scorer from a training output (adapter/, head.pt, run_config.json). Merges LoRA into the bf16
    weights (S1) unless the base is 4-bit. Returns (scorer, run_config, temperature)."""
    from peft import PeftModel
    model_dir = Path(model_dir)
    run = load_json(model_dir / "run_config.json")
    # the recorded path only exists inside the original job; fall back to the Hub id when it is not mounted here
    base = base_model or (run["base_model"] if Path(run["base_model"]).exists() else run.get("base_model_id", run["base_model"]))
    backbone, ple, tcfg = load_backbone(base, device=device, dtype=dtype, load_4bit=load_4bit, attn=attn, ple_device=ple_device)
    if (model_dir / "adapter").exists():
        backbone = PeftModel.from_pretrained(backbone, str(model_dir / "adapter"), is_trainable=False)
        if merge and not load_4bit:
            backbone = backbone.merge_and_unload()
            log("LoRA merged into the base weights (S1)")
    scorer = BlockScorer(backbone, tcfg.hidden_size, ple)
    scorer.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
    scorer.head.to(device)
    scorer.eval()
    t_file = model_dir / "temperature.json"
    temperature = float(load_json(t_file)["temperature"]) if t_file.exists() else 1.0
    return scorer, run, temperature


def make_loader(ds, num_workers: int, pin: bool):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers, pin_memory=pin,
                      persistent_workers=False, prefetch_factor=4 if num_workers > 0 else None)


def to_device(batch: dict, device) -> dict:
    keys = ("input_ids", "attention_mask", "marker_batch_idx", "marker_pos")
    out = {k: batch[k].to(device, non_blocking=True) for k in keys}
    out["input_ids_cpu"] = batch["input_ids"]
    return out


@torch.inference_mode()
def predict_logits(scorer: nn.Module, store: BlockStore, idx=None, *, token_budget: int, device, reverse: bool = False,
                   num_workers: int = 0, amp_dtype=torch.bfloat16, desc: str = "predict") -> list[np.ndarray]:
    """Logits per block (canonical candidate order) for blocks `idx` of `store`, length-sorted token-budget batches."""
    idx = np.arange(len(store)) if idx is None else np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return []
    local = plan_batches(store.n_tokens[idx], token_budget, shuffle=False)
    ds = BatchDataset(store, [idx[b] for b in local], reverse=reverse, with_labels=False)
    was_training = scorer.training
    scorer.eval()
    res: dict[int, np.ndarray] = {}
    bar = pbar(total=len(idx), desc=desc, unit="blk")
    for batch in make_loader(ds, num_workers, device.type == "cuda"):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits = scorer(**to_device(batch, device)).float().cpu().numpy()
        perm = batch["perm"].numpy() if "perm" in batch else None
        pos = 0
        for b, n in zip(batch["block_index"].tolist(), batch["n_markers"].tolist()):
            lg = logits[pos:pos + n]
            if perm is not None:
                canon = np.empty(n, dtype=np.float32)
                canon[perm[pos:pos + n]] = lg
                lg = canon
            res[b] = lg
            pos += n
        bar.update(len(batch["block_index"]))
    bar.close()
    if was_training:
        scorer.train()
    return [res[int(i)] for i in idx]
