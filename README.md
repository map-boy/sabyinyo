# sabyinyo

Training a code generation model from scratch - and via fine-tuning an open base model - for Python, TypeScript, Bash, and English.

Built by Mugisha Alain Paisible (Gold) - VAF UBWENGE TECH.

## What this is

An end-to-end pipeline covering the full path from raw data to a deployable coding assistant:

1. Data collection and cleaning (Python/TypeScript/Bash + docs/Q&A)
2. Tokenizer training (BPE, code-aware)
3. Model architecture (decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA)
4. Pretraining
5. Supervised fine-tuning (instruction following)
6. Preference alignment (DPO)
7. Evaluation (HumanEval-style, TypeScript, Bash)
8. Serving (vLLM OpenAI-compatible endpoint)

Two paths are supported, per the project plan:
- Path A - full pretrain from zero: learning milestone, small model (125M params), run on a single rented GPU.
- Path C - fine-tune an open base model: the model actually used day to day, via LoRA/QLoRA on top of an existing open code model.

## Structure

- configs/    tokenizer, model, and training configs
- data/       raw/processed data and collection/cleaning scripts
- tokenizer/  BPE tokenizer training
- model/      transformer architecture (architecture.py, layers.py, checkpoint_utils.py)
- training/   pretrain.py, finetune.py (SFT), dpo.py, distributed_setup.py, callbacks.py
- eval/       evaluation harness + diagnostics, HumanEval-style runner, TypeScript/Bash syntax checks, smoke test
- inference/  generate.py, policy.py (decision layer), serve.py (vLLM endpoint)
- notebooks/  Colab training and testing notebooks
- docs/       MODEL_SPEC.md (behaviour rules), FINDINGS.md (checkpoint analysis), MVP_ARCHITECTURE.md (build plan)
- .github/    CI workflow (lint, syntax check, smoke test, spec compliance)

## Setup

pip install -r requirements.txt

## CI

Every push runs automatically via GitHub Actions: Python syntax validation, lint (ruff), a model forward-pass smoke test, and the spec-compliance suite for the decision layer.

Check status: https://github.com/map-boy/sabyinyo/actions

## Training

Local dev and CI have no GPU. Actual training runs happen in Google Colab:
1. Open notebooks/train_colab.ipynb
2. It clones this repo, installs deps, and logs into Hugging Face Hub
3. Checkpoints get pushed to the Hugging Face model repo

Model checkpoints: https://huggingface.co/map-boy/sabyinyo-codegen
Datasets: https://huggingface.co/datasets/map-boy/sabyinyo-data

## Testing a checkpoint

Run the full suite against a checkpoint on the Hub:

    PYTHONPATH=. python eval/run_eval.py --data-dir /content/data --checkpoint latest

It downloads the checkpoint, scores held-out perplexity against three baselines
(uniform-random, a unigram table, and an untrained model of the same shape),
runs structural diagnostics, generates samples, and exits non-zero if any check
fails. `notebooks/test_model_colab.ipynb` walks through the same thing
interactively in Colab, with the secrets wiring and a loss-vs-step curve.

Required Colab secrets: `hug_read`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
(`HF_TOKEN_WRITE` only for pushing checkpoints).

## Model behaviour

`docs/MODEL_SPEC.md` is the rulebook: what the model answers, refuses, or asks
about, with the tiers ordered so conflicts resolve the same way every time
(safety > honesty > correctness > compliance > helpfulness).

`inference/policy.py` implements the enforceable rules as two pure functions -
`decide(prompt)` before generation, `validate(text, decision)` after - so
destructive commands, leaked credentials, and unparseable code are caught by
deterministic code rather than hoped away.

    PYTHONPATH=. python eval/behavior_eval.py --policy-only

scores the decision layer against the spec, rule by rule, with no checkpoint
required. It runs in CI. Drop `--policy-only` to also generate with a real
checkpoint and measure how often the output clears the enforced rules.

## Status

Scaffold and CI complete. The first pretraining run (100k configured steps,
checkpoints on the Hub) does not beat a random baseline; `docs/FINDINGS.md` has
the analysis and the fixes, and `docs/MVP_ARCHITECTURE.md` has the plan for the
next attempt.

## License

MIT - see LICENSE.
