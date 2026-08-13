import torch
import torch.nn.functional as F

from model.checkpoint_utils import save_checkpoint


def cross_entropy(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"])
            loss = cross_entropy(logits, batch["labels"])
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def generate(model, prompt, max_tokens=100):
    # placeholder greedy loop; wire up real tokenizer before use
    return prompt


def train(model, optimizer, scheduler, dataloader, val_loader, cfg):
    model.train()
    for step, batch in enumerate(dataloader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["input_ids"])
            loss = cross_entropy(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % cfg.get("eval_every", 1000) == 0:
            val_loss = evaluate(model, val_loader)
            print(f"[step {step}] train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

        if step % cfg.get("save_every", 2000) == 0:
            save_checkpoint(model, optimizer, step)

        if step >= cfg.get("max_steps", 100000):
            break


if __name__ == "__main__":
    pass
