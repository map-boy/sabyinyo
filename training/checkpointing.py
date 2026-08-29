"""Fully resumable training state, checkpointed to the Hugging Face Hub.

The design requirement: a run can start on Kaggle, die (session timeout, quota,
manual stop), and resume later on Colab from the exact recorded step -- training
state, not just weights. So a checkpoint carries everything needed to make the
next optimizer step identical to the one the dead process would have taken:

    model weights, optimizer state, LR scheduler state, global step, epoch,
    the RNG state of python / numpy / torch / cuda, and the resolved config.

Everything here is platform-agnostic and has no import-time dependency on torch,
transformers, or huggingface_hub -- they are imported inside the functions that
need them, so `import training.checkpointing` is cheap and CI can lint it
without a GPU stack installed.
"""

import io
import json
import os
import random
import signal
import tarfile
import tempfile
import time

# One checkpoint is a directory of these files, tarred into a single Hub upload
# so a half-finished push can never leave a checkpoint that loads partially.
STATE_FILE = "training_state.pt"     # optimizer, scheduler, step, epoch, rng
CONFIG_FILE = "resolved_config.json"
META_FILE = "checkpoint_meta.json"


# ---------------------------------------------------------------------------
# RNG state -- the part everyone forgets, and the reason "resume" usually means
# "restart with a fresh data order and a fresh dropout mask".
# ---------------------------------------------------------------------------
def capture_rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ---------------------------------------------------------------------------
# Hub upload/download with retry + backoff
# ---------------------------------------------------------------------------
def _with_backoff(fn, *, what, attempts=5, base=2.0, max_sleep=60.0):
    """Run fn(), retrying transient Hub failures with exponential backoff.

    HF Hub rate-limits (HTTP 429) and has the usual transient 5xx / connection
    resets. Anything else (bad token -> 401/403, missing repo -> 404) is not
    retryable and is re-raised immediately.
    """
    from huggingface_hub.utils import HfHubHTTPError

    last = None
    for i in range(attempts):
        try:
            return fn()
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403, 404):
                raise
            last = e
        except (OSError, ConnectionError) as e:
            last = e
        sleep = min(base ** i, max_sleep)
        print(f"[checkpoint] {what} failed (attempt {i + 1}/{attempts}): "
              f"{type(last).__name__}: {last}. retrying in {sleep:.0f}s")
        time.sleep(sleep)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}")


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------
def save_training_state(out_dir, *, model, optimizer, scheduler, step, epoch,
                        config, is_peft=True):
    """Write a complete, loadable checkpoint directory. Returns its path.

    `model` may be a plain nn.Module or a PEFT-wrapped model. For PEFT we save
    only the adapter (a few MB) via save_pretrained; for a plain module we save
    the whole state_dict. Either way the optimizer/scheduler/step/rng go in
    STATE_FILE so resume is exact.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)

    if is_peft and hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)           # adapter_model.safetensors + config
    else:
        torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "epoch": epoch,
            "rng": capture_rng_state(),
        },
        os.path.join(out_dir, STATE_FILE),
    )
    with open(os.path.join(out_dir, CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump({"step": step, "epoch": epoch, "saved_at": time.time(),
                   "is_peft": is_peft}, f, indent=2)
    return out_dir


def push_checkpoint(out_dir, *, repo_id, token, step, retain=3):
    """Tar the checkpoint dir and upload it to the Hub, then update `latest`.

    Two Hub files change per checkpoint:
      checkpoints/step_<n>.tar.gz  -- the immutable numbered checkpoint
      checkpoints/latest.txt       -- one line naming the newest step, so resume
                                      is a tiny read instead of listing the repo
    Old numbered checkpoints beyond `retain` are deleted to keep the repo small.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _with_backoff(lambda: api.create_repo(repo_id=repo_id, repo_type="model",
                                          exist_ok=True), what="create_repo")

    tar_name = f"step_{step}.tar.gz"
    tar_path = os.path.join(tempfile.gettempdir(), tar_name)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=".")

    _with_backoff(
        lambda: api.upload_file(path_or_fileobj=tar_path,
                                path_in_repo=f"checkpoints/{tar_name}",
                                repo_id=repo_id, repo_type="model"),
        what=f"upload {tar_name}",
    )
    _with_backoff(
        lambda: api.upload_file(path_or_fileobj=io.BytesIO(str(step).encode()),
                                path_in_repo="checkpoints/latest.txt",
                                repo_id=repo_id, repo_type="model"),
        what="upload latest.txt",
    )
    os.remove(tar_path)
    _prune_old_checkpoints(api, repo_id, keep_step=step, retain=retain)
    print(f"[checkpoint] pushed step {step} to {repo_id}")


def _prune_old_checkpoints(api, repo_id, *, keep_step, retain):
    try:
        files = [f for f in api.list_repo_files(repo_id)
                 if f.startswith("checkpoints/step_") and f.endswith(".tar.gz")]
        steps = sorted(int(f.split("step_")[1].split(".tar.gz")[0]) for f in files)
        for old in steps[:-retain]:
            if old == keep_step:
                continue
            api.delete_file(f"checkpoints/step_{old}.tar.gz", repo_id, repo_type="model")
    except Exception as e:  # pruning is best-effort; never fail a run over it
        print(f"[checkpoint] prune skipped: {e}")


# ---------------------------------------------------------------------------
# resuming
# ---------------------------------------------------------------------------
def find_latest_step(repo_id, token):
    """Return the newest checkpoint step on the Hub, or None if the repo is empty.

    Reads checkpoints/latest.txt if present; otherwise falls back to listing.
    Any 'repo does not exist yet' error means a fresh run -> None.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        p = hf_hub_download(repo_id=repo_id, filename="checkpoints/latest.txt", token=token)
        return int(open(p).read().strip())
    except (HfHubHTTPError, ValueError, OSError):
        pass
    try:
        from huggingface_hub import HfApi
        files = [f for f in HfApi(token=token).list_repo_files(repo_id)
                 if f.startswith("checkpoints/step_") and f.endswith(".tar.gz")]
        steps = [int(f.split("step_")[1].split(".tar.gz")[0]) for f in files]
        return max(steps) if steps else None
    except HfHubHTTPError:
        return None


def download_checkpoint(step, repo_id, token, dest_dir):
    """Download and unpack checkpoints/step_<step>.tar.gz into dest_dir."""
    from huggingface_hub import hf_hub_download

    tar_path = _with_backoff(
        lambda: hf_hub_download(repo_id=repo_id,
                               filename=f"checkpoints/step_{step}.tar.gz",
                               token=token),
        what=f"download step_{step}",
    )
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest_dir)   # our own artifact, produced by push_checkpoint
    return dest_dir


def restore_training_state(ckpt_dir, *, optimizer, scheduler):
    """Load optimizer/scheduler/step/epoch/rng from a downloaded checkpoint dir.

    Weights are restored separately by the caller (PEFT adapter vs full module
    load differ), so this handles only the run-continuity state.
    """
    import torch

    state = torch.load(os.path.join(ckpt_dir, STATE_FILE), map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])
    return {"step": state["step"], "epoch": state["epoch"]}


# ---------------------------------------------------------------------------
# graceful interrupt: force one final push before the process dies
# ---------------------------------------------------------------------------
class InterruptGuard:
    """Catch SIGINT/SIGTERM and set a flag the training loop checks each step,
    so the loop can push a final checkpoint at a safe point rather than dying
    mid-optimizer-step with a half-written file.

    Kaggle/Colab preemption arrives as SIGTERM, which this catches. A hard
    SIGKILL cannot be caught by any process -- the periodic safety-net pushes
    (every save_every_min minutes) are what bound loss in that case.

    Usage:
        guard = InterruptGuard()
        for batch in loader:
            train_step(...)
            if guard.triggered:
                checkpoint_now(); break
    """

    def __init__(self):
        self.triggered = False
        self._signal = None
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not in main thread (e.g. some notebook backends)

    def _handle(self, signum, frame):
        self._signal = signum
        self.triggered = True
        print(f"\n[checkpoint] caught signal {signum}; "
              f"will checkpoint and exit at the next safe point.")
