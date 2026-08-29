# sabyinyo-codegen: checkpoint analysis

Analysis of `map-boy/sabyinyo-codegen` at `checkpoints/latest.pt` (step 100005),
and of the code that produced it. Every number below was reproduced by running
the architecture in this repo; the reproduction commands are in
`notebooks/test_model_colab.ipynb` and `eval/run_eval.py`.

## Summary

The checkpoint scores a held-out cross-entropy of ~13.1-13.6 nats. Uniform-random
guessing over a 32,000-token vocabulary scores `ln(32000) = 10.37`. **The model
is measurably worse than random.** That is not undertraining - it means the run
never converged, and adding steps to the current setup will not fix it.

Four independent defects contribute, in descending order of severity.

---

## 1. The architecture has no positional encoding (blocking)

`README.md` advertises RoPE. `GroupedQueryAttention.__init__` accepts `rope=True`
and stores it as `self.rope`. **`forward()` never uses it**, and `CodeGenModel`
adds no learned position embedding either. There is no positional signal
anywhere in the model.

Reproduced (`eval/diagnostics.py::test_positional_encoding_exists`): build a
1-layer instance, feed it two prefixes that are permutations of each other and
end on the same token.

```
1-layer permutation test: max logit diff = 1.9e-06     # float noise, i.e. zero
```

With causal attention and no positional signal, the last position attends over
the same *set* of key/value vectors either way, so the logits are identical. A
12-layer stack shows a small difference (~9.0 on the same test), but that is not
positional encoding - it leaks in because each earlier position's representation
depends on its own causal prefix *set*, and those sets differ. It is a faint
side channel, not order information the model can use.

Confirmed on trained weights too: shuffling a real 128-token prefix moves the
next-token distribution by KL ~= 0.000 while replacing it with different tokens
moves it measurably. The model reads its context as an unordered bag.

Programming languages are *entirely* about order. `a - b` and `b - a`, `f(x)`
and `x(f)` - a bag-of-tokens model cannot distinguish any of them. This alone
caps achievable quality far below usable.

**Fix** (`model/layers.py`) - apply rotary embeddings to `q` and `k`:

```python
def rope_cache(seq_len, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):          # x: (B, H, T, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

then in `GroupedQueryAttention.forward`, after the `view(...).transpose(1, 2)`
calls and before `repeat_interleave`:

```python
if self.rope:
    cos, sin = rope_cache(T, self.head_dim, x.device)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

Cache `cos`/`sin` in a buffer rather than rebuilding them per forward pass once
it works. **This changes what the weights mean, so it requires retraining from
scratch.**

---

## 2. No weight initialisation (blocking)

`CodeGenModel` never calls an init function, so `nn.Embedding` keeps PyTorch's
default `N(0, 1)`. GPT-style models use `N(0, 0.02)` - a 50x difference. The
`lm_head` is tied to that embedding, so the error is squared into the logits.

Reproduced (`eval/diagnostics.py::test_init_scale`), 127M config:

| | as-is | with `normal_(0, 0.02)` |
|---|---|---|
| embedding std | 1.000 | 0.020 |
| logit std at init | 28.0 | 0.56 |
| next-token loss at init | **679.5** | **10.56** |
| argmax == *current* token | **100%** of positions | ~0% |

Two consequences, both visible in the real training curve:

- The run starts at loss ~680 instead of ~10.4 and spends its entire budget
  climbing out. The measured trajectory is
  `599 (step 2k) -> 31 (20k) -> 21.9 (50k) -> 18.3 (80k) -> 13.6 (100k)` -
  monotone descent that never reached the 10.37 floor.
- At init the model predicts the *current* token at 100% of positions: tied
  `N(0,1)` embeddings dot strongly with themselves. That is a deep copy-collapse
  minimum the model has to escape before it can learn anything.

**Fix** - add to the end of `CodeGenModel.__init__`:

```python
        self.apply(self._init_weights)
        # Scale residual output projections by 1/sqrt(2 * n_layers) so the
        # residual stream does not grow with depth (GPT-2 / Llama convention).
        for block in self.blocks:
            for proj in (block.attn.out_proj, block.ffn.w3):
                nn.init.normal_(proj.weight, 0.0, 0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
```

Verify with `eval/diagnostics.py::test_init_scale` - loss at init should land
near `ln(vocab_size)`.

---

## 3. The training loop does ~1,562 optimizer updates, not 100,000

`kaggle_kernel/train.py` increments `step` once per **micro-batch**, but calls
`optimizer.step()` only every `GRAD_ACCUM = 64` of them. Both the step budget
and the LR warmup are measured against that mismatch.

```
MAX_STEPS = 100000 micro-batches / GRAD_ACCUM = 64  ->  1,562 actual updates
WARMUP_STEPS = 2000 scheduler steps                 ->  128,000 micro-batches
```

Warmup alone wants 1.3x the entire configured run. The learning rate therefore
peaked at ~78% of its 3e-4 target and the cosine decay never ran at all. Total
tokens seen is roughly 410M against a Chinchilla-ish 2.5B for a 127M-parameter
model - about 6x short even before the other defects.

This also explains `step_100000.pt` through `step_100005.pt`. Each is a separate
resumed run: `start_step` is restored from the checkpoint, `step >= MAX_STEPS`
fires after a handful of micro-batches, and the accumulation window
(`(step + 1) % 64 == 0`) never closes - so `optimizer.step()` is never called.
Verified: `step_100000.pt` and `latest.pt` are **bit-identical**, max weight
difference exactly `0.0`. Those six checkpoints contain one set of weights.

**Fix** - count optimizer updates, not micro-batches:

```python
    micro_step = 0
    update = start_update
    while update < MAX_UPDATES:
        for batch in train_loader:
            ...                                  # forward + scaled backward
            micro_step += 1
            if micro_step % GRAD_ACCUM:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad(set_to_none=True)
            update += 1
            if update % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, update)
            if update >= MAX_UPDATES:
                break
```

Save `update` (not `step`) in the checkpoint, and restore the scheduler with
`scheduler.last_epoch = update` on resume - currently the LR schedule restarts
from zero on every resume, so each restart re-warms from LR ~= 0.

Also reconsider the budget itself: `MAX_UPDATES = 100000` at
`2 x 2048 x 64 = 262k tokens/update` is 26B tokens, far more than the 178 MB
corpus holds (~50M tokens). The corpus supports roughly 10-20k updates before
repetition dominates. Either collect more data or size the run to the data.

---

## 4. Evaluation and prompting mistakes (not the model's fault)

These do not affect training, but they made the earlier testing misleading.

**Special tokens do not encode atomically.** Loading
`ByteLevelBPETokenizer(vocab.json, merges.txt)` directly gives a tokenizer whose
ByteLevel pre-tokenizer splits `<|python|>` into `<`, `|`, `python`, `|`, `>`.
`token_to_id("<|python|>")` still returns `9`, which makes it easy to miss - you
have to inspect `encode()`. Fix: call `tokenizer.add_special_tokens([...])`
after loading (`eval/harness.py::load_tokenizer` does this).

**Prompts did not match the corpus.** `train.txt` tags files as
`<filename>path</filename>\n<language>python</language>\n`. The `<|python|>`
markers exist in the vocabulary but never appear in the training data, so every
earlier generation test was out-of-distribution. `eval/run_eval.py` prompts in
the corpus format.

**Perplexity was measured wrong in three ways.** 512-token windows instead of
the 2048 the model was trained on; windows starting mid-file with no context, so
the first tokens (unpredictable by construction) dominated short windows; and
scoring `train.txt` from the front - training data, not held-out. `eval/harness.py`
scores 2048-token windows from the tail 0.5% of the corpus (inside the 1% the
training run reserved), and reports loss both over all tokens and over tokens
with at least 256 tokens of context.

**No baselines.** A perplexity number alone is uninterpretable. Every model must
beat uniform-random (`ln(32000) = 10.37`), a unigram frequency table, and a
randomly-initialised model of the same shape. `eval/run_eval.py` reports all
three next to the model.

---

## Recommended order of work

1. **Fix the eval first** - run `eval/run_eval.py` and record the baseline
   numbers. Without it you cannot tell whether any later change helped.
2. **Fix init and the training loop** (items 2 and 3). Cheap, low-risk, and
   independently verifiable via `test_init_scale` and `test_training_budget`.
3. **Add RoPE** (item 1). This forces a retrain from scratch, so do it in the
   same pass as 2 and 3 rather than separately.
4. **Sanity-run before committing to a long job.** Train on ~2M tokens for a few
   hundred updates and confirm held-out loss drops below the unigram baseline.
   If it does not, the long run will not save it. This costs minutes and would
   have caught all three defects above.
5. **Right-size the run to the corpus.** ~50M tokens supports roughly 10-20k
   optimizer updates. A 127M-parameter model wants ~2.5B tokens; either gather
   more data or expect a model that is a learning exercise rather than a usable
   assistant.
