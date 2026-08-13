import torch
import torch.nn.functional as F


def sft_loss(logits, labels, ignore_index=-100):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index
    )


def train_sft(model, optimizer, dataloader, cfg):
    model.train()
    for step, batch in enumerate(dataloader):
        logits = model(batch["input_ids"])
        loss = sft_loss(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"[sft step {step}] loss={loss.item():.4f}")


if __name__ == "__main__":
    pass
