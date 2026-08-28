"""End-to-end evaluation of a sabyinyo checkpoint.

    python eval/run_eval.py --data-dir /content/data --checkpoint checkpoints/latest.pt

Reads HF_TOKEN (or hug_read / HF_TOKEN_WRITE) from the environment. Prints a
PASS/FAIL table, held-out perplexity against three baselines, and samples.
Exits non-zero if any check fails, so it can gate CI.
"""

import argparse
import os
import sys

import torch

from eval import diagnostics as dx
from eval.harness import (
    HF_REPO_ID,
    build_model,
    download_checkpoint,
    generate,
    list_checkpoints,
    load_model,
    load_tokenizer,
    perplexity,
    read_holdout_text,
    read_train_sample,
    timed_generate,
    uniform_baseline,
    unigram_baseline,
)

# Prompts in the corpus's own format: <filename>/<language> headers, which is
# what the model actually saw during training.
PROMPTS = [
    "<filename>utils/math_helpers.py</filename>\n<language>python</language>\n"
    "def fibonacci(n):\n",
    "<filename>src/types.ts</filename>\n<language>typescript</language>\n"
    "interface User {\n  id: number;\n",
    "<filename>scripts/backup.sh</filename>\n<language>bash</language>\n"
    "#!/usr/bin/env bash\nset -euo pipefail\n",
]


def hf_token():
    for k in ("HF_TOKEN", "hug_read", "HF_TOKEN_READ", "HF_TOKEN_WRITE"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def resolve_checkpoint(name, token, repo_id):
    """Accept a local path, a repo-relative name, or 'latest'."""
    if os.path.exists(name):
        return name
    if name in ("latest", "checkpoints/latest.pt"):
        name = "checkpoints/latest.pt"
    elif not name.startswith("checkpoints/"):
        name = f"checkpoints/{name}"
    available = list_checkpoints(token, repo_id)
    if name not in available:
        raise SystemExit(f"{name} not in {repo_id}. Available: {available[:5]} ...")
    return download_checkpoint(name, token, repo_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/content/data",
                    help="directory holding vocab.json, merges.txt, train.txt")
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--compare-checkpoint", default=None,
                    help="second checkpoint, to prove weights actually changed")
    ap.add_argument("--repo-id", default=HF_REPO_ID)
    ap.add_argument("--windows", type=int, default=8,
                    help="number of 2048-token held-out windows to score")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    token = hf_token()
    results = []

    print("=" * 74)
    print("SABYINYO CHECKPOINT EVALUATION")
    print("=" * 74)

    # --- architecture checks: no checkpoint or data needed -----------------
    print("\n[1/6] Architecture and initialisation (no checkpoint needed)")
    results.append(dx.test_positional_encoding_exists(device="cpu"))
    results.append(dx.test_init_scale(device="cpu"))
    results.append(dx.test_training_budget())
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")

    # --- tokenizer ---------------------------------------------------------
    print("\n[2/6] Tokenizer")
    tok = load_tokenizer(args.data_dir)
    print(f"  vocab size: {tok.get_vocab_size()}")
    for r in (dx.test_tokenizer_roundtrip(tok), dx.test_special_tokens(tok)):
        results.append(r)
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")

    # --- checkpoint --------------------------------------------------------
    print("\n[3/6] Checkpoint")
    ckpt_path = resolve_checkpoint(args.checkpoint, token, args.repo_id)
    model, meta = load_model(ckpt_path, device)
    print(f"  step={meta['step']}  params={meta['n_params']/1e6:.1f}M  device={device}")
    r = dx.test_weight_health(model)
    results.append(r)
    print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")

    if args.compare_checkpoint:
        other = resolve_checkpoint(args.compare_checkpoint, token, args.repo_id)
        r = dx.test_checkpoints_are_distinct(
            ckpt_path, other, args.checkpoint, args.compare_checkpoint)
        results.append(r)
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")

    # --- held-out perplexity ----------------------------------------------
    print("\n[4/6] Held-out perplexity (last 0.5% of train.txt -- never trained on)")
    corpus = os.path.join(args.data_dir, "train.txt")
    holdout_text = read_holdout_text(corpus)
    holdout_ids = tok.encode(holdout_text).ids
    print(f"  held-out tokens available: {len(holdout_ids)}")

    ppl = perplexity(model, holdout_ids, device=device, seq_len=args.seq_len,
                     max_windows=args.windows, verbose=True)
    print(f"\n  MODEL      loss={ppl['loss_warm']:.3f}  ppl={ppl['ppl_warm']:.1f}   "
          f"(over {ppl['n_tokens']} tokens in {ppl['n_windows']} windows, "
          f"first 256 tokens/window excluded)")
    print(f"  (all tokens incl. cold-start: loss={ppl['loss_all']:.3f} "
          f"ppl={ppl['ppl_all']:.1f})")

    uni = unigram_baseline(holdout_ids)
    unif = uniform_baseline()
    print(f"  UNIGRAM    loss={uni['loss']:.3f}  ppl={uni['ppl']:.1f}")
    print(f"  UNIFORM    loss={unif['loss']:.3f}  ppl={unif['ppl']:.1f}")

    print("\n  scoring a randomly-initialised model of the same shape for reference...")
    torch.manual_seed(0)
    untrained = build_model(device).eval()
    untrained_ppl = perplexity(untrained, holdout_ids, device=device,
                               seq_len=args.seq_len, max_windows=2)
    print(f"  UNTRAINED  loss={untrained_ppl['loss_warm']:.3f}  "
          f"ppl={untrained_ppl['ppl_warm']:.1f}")
    del untrained

    for r in (dx.test_beats_uniform(ppl),
              dx.test_beats_unigram(ppl, uni),
              dx.test_beats_untrained(model, untrained_ppl, ppl)):
        results.append(r)
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")

    # --- behaviour ---------------------------------------------------------
    print("\n[5/6] Model behaviour")
    for r in (dx.test_output_entropy(model, holdout_ids, device=device),
              dx.test_order_sensitivity_on_real_model(model, holdout_ids, device=device),
              dx.test_corpus_format_matches_prompts(
                  read_train_sample(corpus, 4096), PROMPTS[0])):
        results.append(r)
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")

    # --- generation --------------------------------------------------------
    if not args.skip_generation:
        print("\n[6/6] Generation samples")
        for prompt in PROMPTS:
            greedy = timed_generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, device=device)
            sampled = generate(model, tok, prompt, max_new_tokens=args.max_new_tokens,
                               temperature=0.8, top_k=50, top_p=0.95,
                               repetition_penalty=1.1, device=device, seed=0)
            print(f"\n  --- prompt ---\n{prompt}")
            print(f"  --- greedy ({greedy['tokens_per_second']:.1f} tok/s) ---")
            print("  " + greedy["completion"].replace("\n", "\n  "))
            print("  --- sampled (T=0.8, top-p=0.95, rep-penalty=1.1) ---")
            print("  " + sampled["completion"].replace("\n", "\n  "))
    else:
        print("\n[6/6] Generation skipped")

    # --- summary -----------------------------------------------------------
    failed = [r for r in results if not r["passed"]]
    print("\n" + "=" * 74)
    print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
    for r in failed:
        print(f"  FAIL  {r['name']}")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
