# MVP architecture for an LLM / NLP product

A reference plan for getting from "nothing" to "something people can use",
written against what `sabyinyo` already has. The organising principle: **every
stage ends in a measurable gate, and you do not start the next stage until the
gate passes.** The current checkpoint is the argument for that rule - 100k
configured steps produced a model worse than random, and nothing in the pipeline
was set up to notice.

## The core decision: two tracks, not one

These are different projects with different economics. Run both; do not confuse
them.

| | Track A - pretrain from scratch | Track B - fine-tune an open base |
|---|---|---|
| Purpose | Learning how LLMs work | The thing users actually use |
| Model | 20-125M params, your own architecture | Qwen2.5-Coder-1.5B, StarCoder2-3B, or similar |
| Data needed | 2.5B+ tokens for 125M params | 1k-50k instruction pairs |
| Compute | Weeks of A100 time | Hours on one T4/A10 with LoRA |
| Realistic quality | Produces plausible-looking tokens | Genuinely useful |
| Time to first usable result | Months | Days |

`README.md` already names these Path A and Path C. The mistake so far has been
treating Track A as the product. **Ship Track B, learn on Track A.**

### Sizing Track A to the data you actually have

The corpus is 178 MB / ~50M tokens. Chinchilla-optimal is roughly 20 tokens per
parameter:

| Params | Tokens wanted | Verdict for a 50M-token corpus |
|---|---|---|
| 125M | 2.5B | 50x short - what was attempted |
| 20M | 400M | 8x short |
| **5M** | **100M** | **2x short - trains to a real loss curve in hours** |

A 5-20M model on 50M tokens will not write good code, but it *will* produce a
clean, monotone loss curve that beats every baseline - which is what a learning
milestone is for. Either shrink the model to the corpus or grow the corpus to
~2.5B tokens (The Stack v2, CodeParrot, and similar public sets get you there).

---

## MVP pipeline

Seven stages. Each row's gate is a command that exits non-zero on failure.

```
  data --> tokenizer --> model --> train --> eval --> serve --> feedback
    |          |           |         |        |         |          |
   G1         G2          G3        G4       G5        G6         G7
```

### G1 - Data

Build it, then measure it. `data/scripts/clean.py` already does secret
stripping, syntax validation, minification filtering, and dedup - wire it into a
pipeline that emits a manifest.

- Hold out a **real** validation split up front, by *document*, written to its
  own file. Slicing the last 1% of a concatenated token stream (what `train.py`
  does now) means your validation set is whatever files happened to sort last -
  a different distribution, not a random sample.
- Deduplicate across the split boundary, or held-out loss measures memorisation.
- Record token counts per language. You cannot size the model without them.

**Gate:** `docs/` records total tokens, per-language breakdown, and confirmed
zero document overlap between train and validation.

### G2 - Tokenizer

`tokenizer/train_tokenizer.py` is fine as far as it goes. What is missing is
verification.

- Every special token must encode to exactly **one** id. Loading
  `ByteLevelBPETokenizer(vocab.json, merges.txt)` raw does not do this - the
  ByteLevel pre-tokenizer shreds `<|python|>` into five tokens while
  `token_to_id()` still returns 9, so the bug hides. `eval/harness.py::load_tokenizer`
  registers them properly.
- Encode/decode must round-trip losslessly on code, whitespace, and unicode.
- Check the compression ratio (chars per token). Below ~3 on code means the
  vocabulary is badly fitted and every sequence wastes context.

**Gate:** `eval/diagnostics.py::test_tokenizer_roundtrip` and
`test_special_tokens` both pass.

### G3 - Model

Decoder-only transformer. The pieces in `model/` are the right pieces -
RMSNorm, SwiGLU, GQA - with two things missing that matter more than any of
them:

- **Positional encoding.** `GroupedQueryAttention` stores `self.rope` and never
  applies it. Without it the model is a bag of tokens: shuffling the prefix
  leaves the prediction unchanged (verified, `max logit diff = 1.9e-06` on a
  1-layer instance). Code is entirely about order.
- **Initialisation.** No init function is called, so `nn.Embedding` keeps
  `N(0,1)` instead of `N(0,0.02)`, and the tied `lm_head` squares the error.
  Loss at init is 679.5 instead of 10.56.

Both fixes, with code, are in `docs/FINDINGS.md`. Add a KV cache before serving -
generation is currently quadratic because every new token re-runs the full
forward pass.

**Gate:** `test_positional_encoding_exists` and `test_init_scale` pass; a fresh
model's loss at init is within 10% of `ln(vocab_size)`.

### G4 - Training loop

The single highest-leverage change to how this project runs:

- **Count optimizer updates, not micro-batches.** `train.py` increments `step`
  per micro-batch but steps the optimizer every 64, so `MAX_STEPS = 100000`
  bought 1,562 real updates, and `WARMUP_STEPS = 2000` wanted 128,000
  micro-batches - 1.3x the whole run. The LR never left warmup and cosine decay
  never ran.
- **Restore scheduler state on resume**, or every restart re-warms from LR ~= 0.
- **Overfit one batch first.** Before any long run, train on a single batch until
  loss approaches zero. If it cannot memorise 2048 tokens, it cannot learn 50M.
  This takes two minutes and catches broken init, broken masking, and label
  misalignment.
- **Log to somewhere you will look** - W&B or a CSV. Train loss, val loss, LR,
  grad norm, tokens seen, per update.
- **Alert on the obvious.** Val loss above `ln(vocab_size)` after warmup means
  stop the run; do not discover it 100k steps later.

**Gate:** a 20-minute run on 2M tokens drives held-out loss below the unigram
baseline. If it does not, the long run will not save it.

### G5 - Evaluation

This repo now has the harness: `eval/harness.py`, `eval/diagnostics.py`,
`eval/run_eval.py`, and `notebooks/test_model_colab.ipynb`. The rules it encodes:

- **Never report perplexity without baselines.** Uniform-random
  (`ln(32000) = 10.37`), a unigram frequency table, and a randomly-initialised
  model of the same shape. A number alone is uninterpretable.
- **Score at the training sequence length**, on held-out data, excluding the
  cold-start tokens of each window.
- **Prompt in the corpus's own format.** The corpus tags files
  `<filename>...</filename>` / `<language>...</language>`; prompting with
  `<|python|>` tests out-of-distribution behaviour.
- **Then functional evals**, once perplexity clears the baselines:
  `eval/humaneval_runner.py` (pass@k), `eval/ts_eval.py` (`tsc --noEmit`),
  `eval/bash_eval.py` (`bash -n`). Run generated code in a sandbox, never in
  your training process.

**Gate:** `python eval/run_eval.py` exits 0.

### G5b - Behaviour

Capability and behaviour are separate axes, and they fail separately: a model
that writes correct code and cheerfully emits `rm -rf /` is not shippable.

`docs/MODEL_SPEC.md` states the rules with an explicit precedence order, so
conflicts resolve the same way every time. `inference/policy.py` implements the
enforceable ones as pure functions either side of generation, and
`eval/behavior_eval.py` scores compliance rule by rule.

Worth doing early, not late: the deterministic half (destructive-command
refusal, secret scanning, syntax gating) works regardless of how good the model
is, and it is far cheaper to build before there are users than after.

**Gate:** `python eval/behavior_eval.py --policy-only` exits 0. It needs no
checkpoint, so it gates CI from day one.

### G6 - Serving

Do not build this until G5 passes.

- Track B: vLLM behind an OpenAI-compatible endpoint (`inference/serve.py` has
  the command) - streaming, batching, and KV caching for free.
- Track A: your own model needs a KV cache first, or latency is unusable.
- Put a thin gateway in front: auth, rate limits, request/response logging,
  timeouts, a fallback when the model is down.

**Gate:** p95 latency and tokens/sec measured under realistic concurrency.

### G7 - Feedback

The part that compounds. Log every prompt and completion (with consent), collect
thumbs up/down, and turn accepted completions into the next SFT set. This is
what `training/finetune.py` and `training/dpo.py` are for - DPO needs preference
pairs, and preference pairs come from users.

---

## Suggested repository shape

Mostly what exists, with the gaps named:

```
configs/          model / train / tokenizer YAML  (single source of truth -
                    train.py currently hardcodes copies that have already drifted)
data/scripts/     scrape, clean, tokenize, split   [split is missing]
tokenizer/        BPE training + verification
model/            architecture, layers, checkpoint_utils   [+ kv_cache]
training/         pretrain, finetune (SFT), dpo, callbacks
eval/             harness, diagnostics, run_eval, behavior_eval, functional evals
inference/        generate, policy (decision layer), serve
notebooks/        train_colab, test_model_colab
docs/             MODEL_SPEC, FINDINGS, MVP_ARCHITECTURE
.github/          CI: lint, syntax, smoke test, spec compliance   [+ eval gate]
```

Two structural fixes worth doing early:

1. **Stop duplicating the model in `kaggle_kernel/train.py`.** It inlines
   `layers.py` and `architecture.py` "verbatim" - and they have already diverged
   (the copy pins an SDP backend the original does not). Install the repo as a
   package (`pip install -e .`) and import it. Two copies of a model definition
   is how checkpoints stop loading.
2. **Read hyperparameters from `configs/`** instead of re-declaring them as
   constants. `train_config.yaml` says `batch_size_per_gpu: 8`,
   `precision: bf16`; `train.py` uses 2 and fp16.

---

## Order of work

1. Run `eval/run_eval.py` and record today's numbers. Baseline first.
2. Fix init + the update-counting loop (G3, G4). Cheap, independently verifiable.
3. Add RoPE. Requires a retrain, so batch it with step 2.
4. Overfit-one-batch, then a 20-minute run. Confirm the gate passes.
5. Only then start a long run - right-sized to the corpus (5-20M params) rather
   than 125M.
6. In parallel, start Track B. LoRA on Qwen2.5-Coder-1.5B over your cleaned
   corpus reaches a usable assistant in days, and gives Track A something honest
   to be measured against.
