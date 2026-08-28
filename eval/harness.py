"""Reusable evaluation harness for sabyinyo checkpoints.

Everything here is plain torch + tokenizers + huggingface_hub so it runs
identically in Colab, in a Kaggle kernel, and on a local CPU box.

Typical use:

    from eval.harness import (load_tokenizer, download_checkpoint, load_model,
                              read_holdout_text, perplexity, generate)

    tok = load_tokenizer("/content/data")
    model, meta = load_model(download_checkpoint("checkpoints/latest.pt", token), "cuda")
    ids = tok.encode(read_holdout_text("/content/data/train.txt")).ids
    print(perplexity(model, ids, device="cuda"))
    print(generate(model, tok, "def fibonacci(n):\\n", device="cuda")["completion"])

Or run the whole suite: python eval/run_eval.py --data-dir /content/data
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

from model.architecture import CodeGenModel

# Must match configs/model_config.yaml and kaggle_kernel/train.py.
VOCAB_SIZE = 32000
DIM = 768
N_LAYERS = 12
N_HEADS = 12
SEQ_LEN = 2048

HF_REPO_ID = "map-boy/sabyinyo-codegen"
KAGGLE_DATASET = "mugishaalainpaisible/codegen-corpus-v1"

# train.py holds out the LAST 1% of the token stream as validation and never
# trains on it. We reproduce that split so "held-out" here really is held-out.
VAL_FRACTION = 0.01


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
def load_tokenizer(data_dir):
    """Load the ByteLevelBPE tokenizer and register the special tokens.

    Loading raw vocab.json/merges.txt gives you a tokenizer that does NOT
    recognise "<|python|>" as one token -- the ByteLevel pre-tokenizer splits
    it into "<", "|", "python", "|", ">". add_special_tokens() fixes that.
    """
    from tokenizers import ByteLevelBPETokenizer

    vocab = os.path.join(data_dir, "vocab.json")
    merges = os.path.join(data_dir, "merges.txt")
    for p in (vocab, merges):
        if not os.path.exists(p):
            raise FileNotFoundError(f"tokenizer file missing: {p}")

    tok = ByteLevelBPETokenizer(vocab, merges)
    with open(vocab, encoding="utf-8") as f:
        vocab_map = json.load(f)
    specials = [t for t in SPECIAL_TOKENS if t in vocab_map]
    tok.add_special_tokens(specials)
    return tok


SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<PRE>", "<SUF>", "<MID>",
    "<file_sep>", "<repo_name>",
    "<|python|>", "<|typescript|>", "<|bash|>", "<|english|>",
]


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def list_checkpoints(hf_token, repo_id=HF_REPO_ID):
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    return sorted(f for f in api.list_repo_files(repo_id) if f.startswith("checkpoints/"))


def download_checkpoint(filename, hf_token, repo_id=HF_REPO_ID):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)


def build_model(device="cpu", vocab_size=None, dim=None, n_layers=None,
                n_heads=None, seq_len=None):
    """Instantiate the architecture at the training shape (overridable for tests)."""
    return CodeGenModel(
        vocab_size or VOCAB_SIZE,
        dim or DIM,
        n_layers or N_LAYERS,
        n_heads or N_HEADS,
        seq_len or SEQ_LEN,
    ).to(device)


def load_model(ckpt_path, device="cpu", **shape):
    """Return (model, checkpoint_meta). meta carries the training step."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model("cpu", **shape)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the architecture.\n"
            f"  missing keys:    {sorted(missing)[:8]}\n"
            f"  unexpected keys: {sorted(unexpected)[:8]}\n"
            f"This means model/architecture.py changed since the checkpoint was written."
        )
    model.eval().to(device)
    meta = {
        "step": ckpt.get("step"),
        "path": ckpt_path,
        "n_params": sum(p.numel() for p in model.parameters()),
        "has_optimizer": "optimizer" in ckpt,
    }
    return model, meta


def state_dict_delta(path_a, path_b):
    """Max absolute weight difference between two checkpoints."""
    a = torch.load(path_a, map_location="cpu")["model"]
    b = torch.load(path_b, map_location="cpu")["model"]
    if set(a) != set(b):
        return float("inf"), sorted(set(a) ^ set(b))[:8]
    max_diff = 0.0
    for k in a:
        max_diff = max(max_diff, (a[k].float() - b[k].float()).abs().max().item())
    return max_diff, []


# --------------------------------------------------------------------------
# held-out data
# --------------------------------------------------------------------------
def read_holdout_text(corpus_path, tail_fraction=VAL_FRACTION / 2, max_bytes=8_000_000):
    """Read the tail of train.txt -- the region train.py reserved for validation.

    train.py splits by TOKEN count (last 1%); we take the last 0.5% of BYTES,
    which sits comfortably inside that region regardless of tokens-per-byte.
    Reading only the tail keeps this cheap on a 178 MB corpus.
    """
    size = os.path.getsize(corpus_path)
    start = max(0, size - min(int(size * tail_fraction), max_bytes))
    with open(corpus_path, "rb") as f:
        f.seek(start)
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    # Drop the first partial line so we start on a clean boundary.
    return text.split("\n", 1)[1] if "\n" in text else text


def read_train_sample(corpus_path, max_bytes=2_000_000):
    """Read from the START of the corpus -- data the model definitely trained on."""
    with open(corpus_path, "rb") as f:
        raw = f.read(max_bytes)
    return raw.decode("utf-8", errors="ignore").rsplit("\n", 1)[0]


# --------------------------------------------------------------------------
# perplexity
# --------------------------------------------------------------------------
@torch.no_grad()
def perplexity(model, token_ids, device="cpu", seq_len=SEQ_LEN, stride=None,
               context_warmup=256, max_windows=None, verbose=False):
    """Sliding-window perplexity, reported two ways.

    - `ppl_all`      : every predicted token counted, including the first ones
                       in a window that have almost no context.
    - `ppl_warm`     : only tokens with at least `context_warmup` tokens of
                       context. This is the honest number; the first tokens of
                       a window are unpredictable by construction and inflate
                       `ppl_all` for reasons that have nothing to do with the
                       model's quality.

    Evaluating at `seq_len` (2048) matters: the model was trained on 2048-token
    chunks, so scoring it on 512-token windows measures a different regime.
    """
    stride = stride or seq_len
    # A warmup longer than the window would silently score nothing.
    context_warmup = min(context_warmup, seq_len // 4)
    model.eval()
    sum_all, n_all = 0.0, 0
    sum_warm, n_warm = 0.0, 0
    n_windows = 0

    for start in range(0, max(1, len(token_ids) - 1), stride):
        chunk = token_ids[start:start + seq_len + 1]
        if len(chunk) < context_warmup + 8:
            break
        inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits = model(inp)
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="none"
        )
        sum_all += losses.sum().item()
        n_all += losses.numel()
        warm = losses[context_warmup:]
        sum_warm += warm.sum().item()
        n_warm += warm.numel()
        n_windows += 1
        if verbose:
            print(f"  window@{start}: {losses.numel()} tokens, "
                  f"loss={losses.mean().item():.3f}")
        if max_windows and n_windows >= max_windows:
            break

    if n_windows == 0 or n_warm == 0:
        raise ValueError(
            f"no scorable windows: {len(token_ids)} held-out tokens is not enough "
            f"for seq_len={seq_len} (need > {context_warmup + 8}). "
            f"Lower --seq-len or widen the held-out slice."
        )

    loss_all = sum_all / max(n_all, 1)
    loss_warm = sum_warm / max(n_warm, 1)
    return {
        "loss_all": loss_all,
        "ppl_all": math.exp(min(loss_all, 60)),
        "loss_warm": loss_warm,
        "ppl_warm": math.exp(min(loss_warm, 60)),
        "n_tokens": n_all,
        "n_windows": n_windows,
    }


def uniform_baseline(vocab_size=VOCAB_SIZE):
    """Loss of a model that has learned nothing at all."""
    return {"loss": math.log(vocab_size), "ppl": float(vocab_size)}


def unigram_baseline(token_ids, vocab_size=VOCAB_SIZE):
    """Loss of a model that only learned token frequencies -- no context at all.

    Any language model worth the GPU time must beat this by a wide margin.
    """
    counts = {}
    for t in token_ids:
        counts[t] = counts.get(t, 0) + 1
    total = len(token_ids)
    # Laplace smoothing so unseen tokens do not give infinite loss.
    denom = total + vocab_size
    loss = -sum(
        c * math.log((c + 1) / denom) for c in counts.values()
    ) / total
    return {"loss": loss, "ppl": math.exp(min(loss, 60))}


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8,
             top_k=50, top_p=0.95, repetition_penalty=1.1, device="cpu",
             stop_on_eos=True, seed=None):
    """Sampling with top-k / top-p / repetition penalty.

    NOTE: CodeGenModel has no KV cache, so every new token re-runs the full
    forward pass over the whole prefix. Cost is O(n^2). Keep max_new_tokens
    small on CPU.
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt encoded to zero tokens")
    eos_id = tokenizer.token_to_id("<eos>")
    generated = list(ids)

    for _ in range(max_new_tokens):
        window = generated[-model.max_seq_len:]
        logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1].float()

        if repetition_penalty and repetition_penalty != 1.0:
            for t in set(window):
                logits[t] = logits[t] / repetition_penalty if logits[t] > 0 else logits[t] * repetition_penalty

        if temperature <= 0:
            next_id = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                cut = cum > top_p
                cut[0] = False  # always keep the top token
                logits[sorted_idx[cut]] = float("-inf")
            next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

        generated.append(next_id)
        if stop_on_eos and eos_id is not None and next_id == eos_id:
            break

    return {
        "prompt": prompt,
        "completion": tokenizer.decode(generated[len(ids):]),
        "full": tokenizer.decode(generated),
        "n_prompt_tokens": len(ids),
        "n_new_tokens": len(generated) - len(ids),
    }


@torch.no_grad()
def timed_generate(model, tokenizer, prompt, **kw):
    t0 = time.time()
    out = generate(model, tokenizer, prompt, **kw)
    out["seconds"] = time.time() - t0
    out["tokens_per_second"] = out["n_new_tokens"] / max(out["seconds"], 1e-9)
    return out
