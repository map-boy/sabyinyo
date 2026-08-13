import os

import torch


def save_checkpoint(model, optimizer, step, out_dir="checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step}.pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )
    return path


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)
