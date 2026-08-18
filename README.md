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
- eval/       HumanEval-style runner, TypeScript/Bash syntax checks, smoke test
- inference/  generate.py, serve.py (vLLM endpoint)
- notebooks/  Colab training notebook
- .github/    CI workflow (lint, syntax check, smoke test)

## Setup

pip install -r requirements.txt

## CI

Every push runs automatically via GitHub Actions: Python syntax validation, lint (ruff), and a model forward-pass smoke test.

Check status: https://github.com/map-boy/sabyinyo/actions

## Training

Local dev and CI have no GPU. Actual training runs happen in Google Colab:
1. Open notebooks/train_colab.ipynb
2. It clones this repo, installs deps, and logs into Hugging Face Hub
3. Checkpoints get pushed to the Hugging Face model repo

Model checkpoints: https://huggingface.co/map-boy/sabyinyo-codegen
Datasets: https://huggingface.co/datasets/map-boy/sabyinyo-data

## Status

Scaffold and CI complete. Data collection (Phase 1) is next.

## License

MIT - see LICENSE.
