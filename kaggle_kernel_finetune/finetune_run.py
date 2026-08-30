import os
import subprocess
import sys

token_path = "/kaggle/input/hf-secret/hf_token_write.txt"
with open(token_path) as f:
    os.environ["HF_TOKEN_WRITE"] = f.read().strip()

subprocess.run(["git", "clone", "https://github.com/map-boy/sabyinyo.git"], check=True)
os.chdir("sabyinyo")

# Kaggle's preinstalled torch (currently 2.10.0+cu128) has DROPPED sm_60
# (Pascal / P100) kernels -- "Minimum ... cuda capability supported ... (7.0)".
# Every P100 run dies at the first embedding lookup with:
#   torch.AcceleratorError: CUDA error: no kernel image is available
#     for execution on the device
# kaggle_kernel/train.py (the Path A pretrain script) already solved this
# exact problem the same way: force-reinstall a torch build old enough to
# still ship sm_60 kernels, before importing torch anywhere else.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "--force-reinstall",
    "torch==2.4.1",
    "--extra-index-url", "https://download.pytorch.org/whl/cu121",
], check=True)

# Pinning torch alone leaves Kaggle's preinstalled torchvision/torchaudio
# (compiled against torch 2.10.0+cu128) mismatched against the torch 2.4.1
# just installed above -- their compiled ops (e.g. torchvision::nms) no
# longer resolve. `transformers`/`peft` touch torchvision during import even
# for a pure text model, so the mismatch surfaces as a RuntimeError that
# transformers' lazy-import machinery re-wraps into a misleading
# "ModuleNotFoundError: Could not import module 'BloomPreTrainedModel'".
#
# peft.get_peft_model() separately checks torchao's version whenever torchao
# is importable at all -- Kaggle preinstalls torchao 0.10.0, which fails
# peft's internal ">=0.16.0" floor:
#   ImportError: Found an incompatible version of torchao. Found version
#     0.10.0, but only versions above 0.16.0 are supported
#
# None of the three are used by this run -- LoRA in fp16, no quantization
# config references torchao, no vision code path exists in a text model --
# so the fix for all three is the same: remove them rather than chase a
# version pin that would need to satisfy both peft's floor AND torch 2.4.1's
# compiled-kernel ABI simultaneously. `is_torchvision_available()` /
# `is_torchao_available()` correctly treat "not installed" as false and skip
# those code paths; they only crash on "installed but incompatible".
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y",
                "torchvision", "torchaudio", "torchao"], check=False)

# The documented install path (see pyproject.toml / README): the finetune
# extra pulls the exact transformers/peft/accelerate/bitsandbytes versions
# training/finetune.py was written against. The previous version of this
# script used `pip install -r requirements.txt` instead -- that file is the
# LEGACY Path A list and does not include peft at all; it only "worked"
# because Kaggle's base image happens to preinstall a peft build. Installing
# it explicitly removes that silent dependency on the platform image.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[finetune]"], check=True)

# --no-resume (temporary): the last run (v23) trained loss=nan the entire
# way through -- fp16 with no loss scaling -- and pushed NaN-poisoned
# checkpoints to map-boy/sabyinyo-codegen, up through checkpoints/latest.txt.
# Without this flag, this run's resume-on-startup would silently pick that
# up and continue from garbage weights instead of starting clean now that
# the GradScaler fix (training/finetune.py) is in. Remove --no-resume once
# a run with finite loss has pushed a good checkpoint, so future runs go
# back to resuming normally -- that's the entire point of the checkpointing
# system, and this repo does not currently expose a way to delete the
# poisoned Hub checkpoints without a write token.
# PYTHONUNBUFFERED: without it, stdout is block-buffered for a non-TTY
# child process. print() calls made once before a long GPU-bound stretch
# (e.g. a diagnostic check right after loading a 1.5B-param model) can sit
# in that buffer through the whole stretch and never appear in the log --
# observed directly: [diag] prints placed before model-loading vanished
# entirely from a completed run's log, while prints inside the tight
# per-step training loop (which flush the buffer through sheer volume)
# came through fine. This forces every print, everywhere in this process
# and the modules it imports, to flush immediately.
subprocess.run([sys.executable, "-m", "training.finetune",
                "--config", "configs/finetune_config.yaml",
                "--no-resume"], check=True,
               env={**os.environ, "PYTHONUNBUFFERED": "1"})