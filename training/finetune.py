"""Path C: resumable LoRA/QLoRA fine-tune of an open code base model.

This REPLACES the earlier stub. The previous training/finetune.py was a
15-line sft_loss + a toy loop over CodeGenModel-style logits that nothing in
the repo imported; it could not fine-tune an HF base model and had no resume.
This module does the real job:

  clone-and-run: `python -m training.finetune --config configs/finetune_config.yaml`
  resume:        automatic. On startup it checks the HF repo for the latest
                 checkpoint and, if one exists, restores weights + optimizer +
                 scheduler + step + RNG and continues. No manual diffing.
  survives:      a periodic safety-net push (every N steps or T minutes) plus a
                 SIGINT/SIGTERM handler that forces one final push before exit,
                 so an interrupted run loses at most a few minutes.

Heavy deps (torch, transformers, peft, datasets) are imported inside main() so
that importing this module for its helpers -- or linting it in CI without a GPU
stack -- stays cheap.
"""

import argparse
import math
import os
import sys

from training import checkpointing as ckpt


def load_config(path):
    import yaml

    with open(path, encoding="utf-8-sig") as f:   # tolerate a UTF-8 BOM
        return yaml.safe_load(f)


def hf_token():
    """A write token is required to push checkpoints."""
    for k in ("HF_TOKEN_WRITE", "HF_TOKEN", "hug_write"):
        if os.environ.get(k):
            return os.environ[k]
    raise SystemExit(
        "No HF write token in env (HF_TOKEN_WRITE). Fine-tuning pushes "
        "checkpoints, so a write token is required."
    )


def build_dataset(cfg, tokenizer):
    """Return a list of {input_ids, labels} with the prompt masked out of the
    loss (labels = -100 on prompt tokens), so SFT trains only on responses.
    """
    from datasets import load_dataset

    path = cfg["dataset"]
    if os.path.exists(path):
        ds = load_dataset("json", data_files=path, split="train")
    else:
        ds = load_dataset(path, split="train")

    tmpl = cfg["prompt_template"]
    max_len = cfg["max_seq_len"]

    def encode(row):
        prompt = tmpl.split("{response}")[0].format(prompt=row["prompt"], response="")
        full = tmpl.format(prompt=row["prompt"], response=row["response"])
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
        labels = list(f_ids)
        for i in range(min(len(p_ids), len(labels))):
            labels[i] = -100                       # mask the prompt
        return {"input_ids": f_ids, "labels": labels}

    return ds.map(encode, remove_columns=ds.column_names)


def collate(batch, pad_id):
    import torch

    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, lbls, mask = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        lbls.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(lbls),
        "attention_mask": torch.tensor(mask),
    }


def build_model_and_tokenizer(cfg):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"],
                                        trust_remote_code=cfg["trust_remote_code"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # attn_implementation deliberately left at the transformers default (NOT
    # forced to "eager"). This repo's one and only Path C run that produced
    # real, finite, converging loss (1.19 -> 0.33 -> 0.0001 over 200 steps,
    # 8/29 ~17:59) used the default. "eager" was added afterward, ~19:08,
    # while chasing a separate issue -- every run since, on this exact GPU
    # and dataset, has trained loss=nan from step 0 with eager forced. If a
    # future run needs eager again for some other reason, that regression
    # needs to be reproduced and understood first, not silently reintroduced.
    if cfg["precision"] == "bf16":
        model_dtype = torch.bfloat16
    elif cfg["precision"] == "fp32":
        model_dtype = torch.float32
    else:
        model_dtype = torch.float16
    kwargs = {"trust_remote_code": cfg["trust_remote_code"], "torch_dtype": model_dtype}
    if cfg["load_in_4bit"]:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_config.yaml")
    ap.add_argument("--out-dir", default=None, help="override cfg out_dir")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any Hub checkpoint and start fresh")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    token = hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = cfg["hf_repo_id"]

    torch.manual_seed(cfg["seed"])
    model, tok = build_model_and_tokenizer(cfg)
    model.to(device)

    # DIAGNOSTIC (temporary): loss=nan reproduces identically in fp16 and
    # fp32, with a fresh (non-resumed) model, ruling out precision, attn
    # implementation, and resumed-from-poisoned-checkpoint contamination.
    # This pinpoints whether the fault is in the loaded weights themselves
    # or in the label/logit path, before spending a step on it.
    n_nonfinite = sum((~torch.isfinite(p)).sum().item() for p in model.parameters())
    print(f"[diag] non-finite values across all parameters: {n_nonfinite}")

    ds = build_dataset(cfg, tok)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )
    # autocast+GradScaler only make sense for fp16 mixed precision. Neither
    # is the amp/fp16 path itself: `precision: fp32` is a diagnostic fallback
    # for isolating whether a numerical fault is fp16-specific -- with it,
    # every op runs at full precision and both are disabled outright.
    use_amp = cfg["precision"] not in ("bf16", "fp32") and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = cfg["max_steps"]
    warmup = int(cfg["warmup_ratio"] * total)

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        prog = (s - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- resume-on-startup ------------------------------------------------
    start_step = 0
    if not args.no_resume:
        latest = ckpt.find_latest_step(repo_id, token)
        if latest is not None:
            print(f"[resume] found checkpoint at step {latest}; restoring")
            d = os.path.join(cfg["out_dir"], "_resume")
            ckpt.download_checkpoint(latest, repo_id, token, d)
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.get_base_model(), d, is_trainable=True)
            model.to(device)
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            st = ckpt.restore_training_state(d, optimizer=optimizer, scheduler=scheduler)
            start_step = st["step"]
            print(f"[resume] continuing from step {start_step}")
        else:
            print("[resume] no checkpoint on the Hub; starting fresh")

    guard = ckpt.InterruptGuard()

    def checkpoint(step):
        d = os.path.join(cfg["out_dir"], f"step_{step}")
        ckpt.save_training_state(d, model=model, optimizer=optimizer,
                                 scheduler=scheduler, step=step, epoch=0,
                                 config=cfg, is_peft=True)
        ckpt.push_checkpoint(d, repo_id=repo_id, token=token, step=step,
                             retain=cfg["retain_checkpoints"])

    import time as _t
    model.train()
    step = start_step
    last_save = _t.time()
    optimizer.zero_grad()
    done = False
    while step < total and not done:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
                loss = out.loss / cfg["grad_accum"]

            if step == start_step:
                n_valid_labels = (batch["labels"] != -100).sum().item()
                n_zero_attn_rows = (batch["attention_mask"].sum(dim=1) == 0).sum().item()
                print(f"[diag] first batch: valid (non -100) label count={n_valid_labels}, "
                      f"rows with all-zero attention_mask={n_zero_attn_rows}, "
                      f"logits non-finite={(~torch.isfinite(out.logits)).sum().item()}, "
                      f"logits shape={tuple(out.logits.shape)}")
            scaler.scale(loss).backward()
            if (step + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % 20 == 0:
                print(f"[step {step}/{total}] loss={out.loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            due = (step > start_step and step % cfg["save_every_steps"] == 0) or \
                  (_t.time() - last_save >= cfg["save_every_min"] * 60)
            if due:
                checkpoint(step)
                last_save = _t.time()

            if guard.triggered:
                print("[checkpoint] interrupt -> final push")
                checkpoint(step)
                done = True
                break

            step += 1
            if step >= total:
                done = True
                break

    checkpoint(step)
    print(f"[done] training complete at step {step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
