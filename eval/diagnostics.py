"""Diagnostic tests that explain *why* a checkpoint is good or bad.

Perplexity alone tells you the model is bad. These tests tell you which of the
usual suspects is responsible: the architecture, the initialisation, the
tokenizer, the training loop, or simply not enough optimizer updates.

Every function returns a dict with at least {"name", "passed", "detail"} so
run_eval.py can print a table and a script can assert on it.
"""

import math

import torch
import torch.nn.functional as F

from eval import harness as H
from eval.harness import SPECIAL_TOKENS, build_model, state_dict_delta, uniform_baseline
from model.architecture import CodeGenModel


def _result(name, passed, detail, **extra):
    out = {"name": name, "passed": passed, "detail": detail}
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# 1. Architecture: does the model encode token ORDER at all?
# --------------------------------------------------------------------------
def test_positional_encoding_exists(device="cpu"):
    """A decoder-only transformer with no positional encoding is a bag of tokens.

    Proof by construction: build a 1-layer instance of this exact architecture
    and feed it two prefixes that are permutations of each other, ending on the
    same token. With causal attention and no positional signal, the last
    position attends over the same *set* of key/value vectors in both cases, so
    the logits are bit-identical. If the architecture had RoPE (or learned
    positions), they would differ substantially.
    """
    torch.manual_seed(0)
    m = CodeGenModel(vocab_size=256, dim=64, n_layers=1, n_heads=4, max_seq_len=64)
    m.eval().to(device)
    a = torch.tensor([[11, 22, 33, 44, 55, 7]], device=device)
    b = torch.tensor([[44, 11, 55, 22, 33, 7]], device=device)  # same multiset, same last token
    with torch.no_grad():
        diff = (m(a)[0, -1] - m(b)[0, -1]).abs().max().item()

    passed = diff > 1e-3
    return _result(
        "positional-encoding present",
        passed,
        (
            f"1-layer permutation test: max logit diff = {diff:.2e}. "
            + ("Order affects the output -> positions are encoded."
               if passed else
               "Shuffling the prefix changes NOTHING. The architecture has no "
               "positional encoding: GroupedQueryAttention stores self.rope but "
               "never applies it, and CodeGenModel adds no position embedding. "
               "The model cannot learn syntax, which is entirely about order.")
        ),
        max_logit_diff=diff,
    )


def test_order_sensitivity_on_real_model(model, token_ids, device="cpu", n_trials=8, seed=0):
    """How much does a *trained* model care about prefix order, in practice?

    Compares two perturbations of the same prefix:
      - SHUFFLE : same tokens, different order  (tests order sensitivity)
      - REPLACE : different tokens entirely     (tests content sensitivity)
    Both measured as KL from the unperturbed next-token distribution.

    A healthy LM has shuffle_kl of the same order as replace_kl. shuffle_kl
    near zero means the model is reading its context as an unordered bag.
    """
    g = torch.Generator().manual_seed(seed)
    model.eval()
    shuffle_kls, replace_kls = [], []

    with torch.no_grad():
        for i in range(n_trials):
            start = (i * 977) % max(1, len(token_ids) - 200)
            base = token_ids[start:start + 128]
            if len(base) < 32:
                continue

            def dist(ids):
                logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1].float()
                return F.log_softmax(logits, dim=-1)

            p = dist(base)

            perm = torch.randperm(len(base) - 1, generator=g).tolist()
            shuffled = [base[j] for j in perm] + [base[-1]]
            replaced = torch.randint(0, H.VOCAB_SIZE, (len(base) - 1,), generator=g).tolist() + [base[-1]]

            shuffle_kls.append(F.kl_div(dist(shuffled), p, log_target=True, reduction="sum").abs().item())
            replace_kls.append(F.kl_div(dist(replaced), p, log_target=True, reduction="sum").abs().item())

    s = sum(shuffle_kls) / max(len(shuffle_kls), 1)
    r = sum(replace_kls) / max(len(replace_kls), 1)
    ratio = s / max(r, 1e-9)
    ignores_context = r < 1e-3
    passed = (not ignores_context) and ratio > 0.15

    if ignores_context:
        verdict = (
            "Neither reordering NOR replacing the context changes the prediction: "
            "the model ignores its context entirely and predicts from the last "
            "token alone."
        )
    elif passed:
        verdict = "Order carries real signal."
    else:
        verdict = (
            "Reordering the context barely moves the prediction while replacing it "
            "moves it a lot -> the model is reading context as a bag of tokens."
        )

    return _result(
        "model uses token order",
        passed,
        f"mean KL(shuffled||orig)={s:.3f}, mean KL(replaced||orig)={r:.3f}, "
        f"ratio={ratio:.3f}. " + verdict,
        shuffle_kl=s, replace_kl=r, ratio=ratio,
    )


# --------------------------------------------------------------------------
# 2. Initialisation / weight scale
# --------------------------------------------------------------------------
def test_init_scale(device="cpu", **shape):
    """CodeGenModel applies no custom init, so nn.Embedding keeps its N(0,1)
    default. The lm_head is tied to that embedding, so logits at step 0 have a
    std of ~28 instead of ~0.5, and the initial loss is ~680 instead of ~10.4.

    Two consequences, both visible in the real training curve:
      - the run spends its whole budget climbing out of that hole;
      - argmax at init is the *current* token (tied huge embeddings dot with
        themselves), a copy-collapse minimum the model then has to escape.
    """
    torch.manual_seed(0)
    m = build_model(device, **shape)
    x = torch.randint(0, H.VOCAB_SIZE, (1, 65), device=device)
    with torch.no_grad():
        logits = m(x[:, :-1]).float()
        loss = F.cross_entropy(logits.reshape(-1, H.VOCAB_SIZE), x[:, 1:].reshape(-1)).item()
        self_copy = (logits.argmax(-1) == x[:, :-1]).float().mean().item()

    embed_std = m.embed.weight.std().item()
    uniform = uniform_baseline(H.VOCAB_SIZE)["loss"]
    passed = loss < uniform * 1.5

    return _result(
        "sane initialisation",
        passed,
        f"freshly-initialised model: embed std={embed_std:.3f} (GPT-style is 0.02), "
        f"logit std={logits.std().item():.1f}, loss at init={loss:.1f} "
        f"(uniform baseline is {uniform:.2f}), argmax==current-token at "
        f"{self_copy:.0%} of positions. "
        + ("Init is in the normal range."
           if passed else
           "Init loss is ~65x the uniform baseline. Add normal_(0, 0.02) init "
           "for every Linear/Embedding (and 0.02/sqrt(2*n_layers) on residual "
           "output projections)."),
        embed_std=embed_std, init_loss=loss, self_copy_rate=self_copy,
    )


def test_weight_health(model):
    """NaN/Inf sweep plus per-tensor scale, on a trained checkpoint."""
    bad, scales = [], {}
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
        scales[name] = p.float().abs().max().item()
    biggest = sorted(scales.items(), key=lambda kv: -kv[1])[:3]
    passed = not bad
    return _result(
        "weights finite",
        passed,
        ("no NaN/Inf. " if passed else f"NON-FINITE weights in {bad[:5]}. ")
        + "largest |w|: " + ", ".join(f"{k}={v:.1f}" for k, v in biggest),
        nonfinite=bad,
    )


# --------------------------------------------------------------------------
# 3. Tokenizer
# --------------------------------------------------------------------------
def test_tokenizer_roundtrip(tokenizer, samples=None):
    """Encode -> decode must be lossless, or your perplexity is measuring noise."""
    samples = samples or [
        "def add(a, b):\n    return a + b\n",
        "interface User {\n  id: number;\n}\n",
        "for i in $(seq 1 10); do echo $i; done\n",
        "  \t mixed   whitespace \n\n",
        "unicode: café → ✓",
    ]
    failures = []
    for s in samples:
        got = tokenizer.decode(tokenizer.encode(s).ids)
        if got != s:
            failures.append((s, got))
    passed = not failures
    return _result(
        "tokenizer round-trips",
        passed,
        "all samples survive encode/decode" if passed
        else f"{len(failures)}/{len(samples)} samples changed, e.g. {failures[0][0]!r} -> {failures[0][1]!r}",
        failures=failures,
    )


def test_special_tokens(tokenizer):
    """Each special token must encode to exactly ONE id.

    Loading ByteLevelBPETokenizer straight from vocab.json/merges.txt does NOT
    do this: the ByteLevel pre-tokenizer shreds "<|python|>" into
    "<", "|", "python", "|", ">". token_to_id() still returns 9, which makes
    the problem easy to miss -- you have to check encode().
    """
    rows, broken = [], []
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        ids = tokenizer.encode(tok).ids
        ok = tid is not None and ids == [tid]
        rows.append((tok, tid, ids, ok))
        if not ok:
            broken.append(tok)
    passed = not broken
    return _result(
        "special tokens encode atomically",
        passed,
        "every special token encodes to a single id" if passed
        else (f"{len(broken)} special tokens get split by the pre-tokenizer, e.g. "
              f"{broken[0]!r} -> {tokenizer.encode(broken[0]).ids}. "
              "Call tokenizer.add_special_tokens([...]) after loading "
              "(eval.harness.load_tokenizer does this)."),
        rows=rows, broken=broken,
    )


def test_corpus_format_matches_prompts(corpus_sample, prompt):
    """Prompts must look like the training data, or you are testing OOD behaviour.

    The corpus uses <filename>...</filename> / <language>...</language> headers.
    Prompting with <|python|> -- which appears in the vocab but not in the
    corpus -- asks the model for something it has never seen.
    """
    head = corpus_sample[:400]
    corpus_uses_lang_tag = "<language>" in head or "<filename>" in head
    prompt_uses_pipe = "<|" in prompt
    passed = not (corpus_uses_lang_tag and prompt_uses_pipe)
    return _result(
        "prompt format matches corpus",
        passed,
        f"corpus starts with: {head[:80]!r}. "
        + ("prompt format is consistent with the corpus."
           if passed else
           "The corpus tags files with <filename>/<language> but the prompt uses "
           "<|lang|> markers that never appear in training data. Prompt with the "
           "corpus format instead."),
    )


# --------------------------------------------------------------------------
# 4. Is the model actually learning? (comparisons, not absolutes)
# --------------------------------------------------------------------------
def test_beats_uniform(ppl_result):
    """The floor no language model may sit below."""
    base = uniform_baseline(H.VOCAB_SIZE)
    loss = ppl_result["loss_warm"]
    passed = loss < base["loss"]
    return _result(
        "beats uniform-random",
        passed,
        f"held-out loss={loss:.3f} (ppl {ppl_result['ppl_warm']:.1f}) vs "
        f"uniform loss={base['loss']:.3f} (ppl {base['ppl']:.0f}). "
        + ("Above the floor." if passed else
           "The model is WORSE than guessing uniformly at random over the vocab. "
           "This is not 'undertrained' -- it means training never converged."),
    )


def test_beats_unigram(ppl_result, unigram):
    """The floor a model with any context modelling must clear."""
    loss = ppl_result["loss_warm"]
    passed = loss < unigram["loss"]
    return _result(
        "beats unigram baseline",
        passed,
        f"held-out loss={loss:.3f} vs unigram loss={unigram['loss']:.3f} "
        f"(ppl {unigram['ppl']:.1f}). "
        + ("The model uses context." if passed else
           "A table of token frequencies predicts this corpus better than the "
           "model does. Nothing about context has been learned."),
    )


def test_beats_untrained(model, untrained_ppl, trained_ppl):
    """Training must beat a randomly-initialised model of the same shape."""
    passed = trained_ppl["loss_warm"] < untrained_ppl["loss_warm"]
    return _result(
        "beats its own random init",
        passed,
        f"trained loss={trained_ppl['loss_warm']:.3f} vs "
        f"untrained loss={untrained_ppl['loss_warm']:.3f}. "
        + ("Training moved the needle." if passed else
           "The checkpoint is no better than random weights."),
    )


def test_output_entropy(model, token_ids, device="cpu", n=4):
    """Degenerate models are either near-deterministic or near-uniform."""
    ents, self_copy = [], []
    with torch.no_grad():
        for i in range(n):
            start = (i * 613) % max(1, len(token_ids) - 300)
            ids = token_ids[start:start + 256]
            if len(ids) < 32:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(inp)[0].float()
            probs = torch.softmax(logits, dim=-1)
            ents.append(float(-(probs * torch.log(probs + 1e-12)).sum(-1).mean()))
            self_copy.append(float((logits.argmax(-1) == inp[0]).float().mean()))
    ent = sum(ents) / max(len(ents), 1)
    copy = sum(self_copy) / max(len(self_copy), 1)
    max_ent = math.log(H.VOCAB_SIZE)
    passed = 0.5 < ent < max_ent * 0.95 and copy < 0.5
    return _result(
        "output distribution is non-degenerate",
        passed,
        f"mean predictive entropy={ent:.3f} nats (uniform would be {max_ent:.2f}), "
        f"argmax==current-token at {copy:.0%} of positions. "
        + ("Distribution looks healthy." if passed else
           "Degenerate: " + ("collapsed onto copying the input token. "
                             if copy >= 0.5 else "entropy is out of the sane band.")),
        entropy=ent, self_copy_rate=copy,
    )


def test_checkpoints_are_distinct(path_a, path_b, label_a="A", label_b="B"):
    """Two checkpoints from different steps must have different weights.

    If they are bit-identical, no optimizer step happened between them. With
    GRAD_ACCUM=64, a resumed run that executes fewer than 64 micro-steps
    before hitting MAX_STEPS never calls optimizer.step() at all.
    """
    diff, key_mismatch = state_dict_delta(path_a, path_b)
    passed = diff > 0.0
    return _result(
        "checkpoints differ",
        passed,
        f"max |{label_a} - {label_b}| = {diff}. "
        + ("Weights moved." if passed else
           "The two checkpoints are BIT-IDENTICAL. The later run resumed, "
           "burned a few micro-steps, and exited before the gradient "
           "accumulation window (64) ever closed, so optimizer.step() was "
           "never called. Those extra checkpoint files contain no new training."),
        max_diff=diff, key_mismatch=key_mismatch,
    )


# --------------------------------------------------------------------------
# 5. Training-budget arithmetic (no GPU needed)
# --------------------------------------------------------------------------
def test_training_budget(max_steps=100000, grad_accum=64, batch_size=2,
                         seq_len=2048, warmup_steps=2000, n_params=127_200_000):
    """train.py counts MICRO-batches, but calls scheduler.step() per UPDATE.

    Both the step budget and the LR warmup are measured against that mismatch,
    so the run does far fewer real updates than the config implies and never
    leaves warmup.
    """
    updates = max_steps // grad_accum
    tokens = updates * grad_accum * batch_size * seq_len
    warmup_micro = warmup_steps * grad_accum
    lr_frac = min(1.0, updates / warmup_steps)
    chinchilla = n_params * 20

    passed = updates >= warmup_steps and tokens >= chinchilla
    return _result(
        "training budget is adequate",
        passed,
        f"MAX_STEPS={max_steps} micro-batches / GRAD_ACCUM={grad_accum} "
        f"= {updates} actual optimizer updates. "
        f"Warmup wants {warmup_steps} updates = {warmup_micro} micro-steps, "
        f"which is {warmup_micro/max_steps:.1f}x the whole budget, so LR peaked at "
        f"{lr_frac:.0%} of its target and cosine decay never ran. "
        f"Tokens seen ~= {tokens/1e6:.0f}M vs a Chinchilla-ish "
        f"{chinchilla/1e9:.1f}B for {n_params/1e6:.0f}M params.",
        updates=updates, tokens=tokens, lr_fraction_of_peak=lr_frac,
    )
