# sabyinyo

Training a code generation model from scratch (and via fine-tuning) for
Python, TypeScript, Bash, and English.

## Structure
- `configs/` — tokenizer, model, and training configs
- `data/` — raw/processed data and collection/cleaning scripts
- `tokenizer/` — BPE tokenizer training
- `model/` — transformer architecture
- `training/` — pretraining, SFT, DPO, distributed setup
- `eval/` — HumanEval-style, TypeScript, and Bash evaluation
- `inference/` — generation and serving

## Status
Scaffold in place. Data collection (Phase 1) is next.
