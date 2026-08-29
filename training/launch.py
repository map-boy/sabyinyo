"""Platform-agnostic launcher: clone -> install -> HF login -> resume -> train.

Both notebooks/train_colab.ipynb and notebooks/train_kaggle.ipynb call this so
that opening either fresh "just continues" a run. The only per-platform
difference is where secrets come from and where output goes; detect_platform()
handles that, so the notebook cell is identical on both.

    from training.launch import launch
    launch()          # reads configs/finetune_config.yaml, resumes if a
                      # checkpoint exists on the Hub, otherwise starts fresh
"""

import os
import subprocess
import sys


def detect_platform():
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    if os.path.isdir("/content"):
        return "colab"
    return "local"


def default_out_dir(platform):
    return {"kaggle": "/kaggle/working/checkpoints",
            "colab": "/content/checkpoints"}.get(platform, "./checkpoints")


def load_secret(name, *aliases):
    """Read a secret from the environment, or the platform's secret store.

    Kaggle: UserSecretsClient. Colab: google.colab.userdata. Falls back to env
    so a local run works with plain environment variables.
    """
    for key in (name, *aliases):
        if os.environ.get(key):
            return os.environ[key]

    platform = detect_platform()
    if platform == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            for key in (name, *aliases):
                try:
                    return client.get_secret(key)
                except Exception:
                    continue
        except Exception:
            pass
    elif platform == "colab":
        try:
            from google.colab import userdata
            for key in (name, *aliases):
                try:
                    v = userdata.get(key)
                    if v:
                        return v
                except Exception:
                    continue
        except Exception:
            pass
    return None


def ensure_repo(repo_url="https://github.com/map-boy/sabyinyo.git", branch="main"):
    """Clone the repo if we're not already inside it, and return its path."""
    if os.path.exists("training/launch.py"):
        return os.getcwd()
    dest = os.path.join(os.getcwd(), "sabyinyo")
    if not os.path.exists(dest):
        subprocess.run(["git", "clone", "--branch", branch, repo_url, dest], check=True)
    os.chdir(dest)
    return dest


def install_deps():
    """Install the package + fine-tune extras WITHOUT touching the platform torch.

    Colab/Kaggle ship a CUDA-matched torch; pyproject.toml deliberately omits
    torch so `pip install -e .` never replaces it. We install the finetune
    extra (transformers/peft/datasets/accelerate/bitsandbytes) on top.
    """
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[finetune]"],
                   check=True)


def launch(config="configs/finetune_config.yaml", branch="main"):
    ensure_repo(branch=branch)
    install_deps()

    platform = detect_platform()
    print(f"[launch] platform = {platform}")

    # Secrets -> environment, so training/finetune.py picks them up uniformly.
    write = load_secret("HF_TOKEN_WRITE", "HF_TOKEN", "hug_write")
    if write:
        os.environ["HF_TOKEN_WRITE"] = write
    else:
        raise SystemExit("No HF write token found (HF_TOKEN_WRITE) in secrets/env.")
    admin = load_secret("SABYINYO_ADMIN_TOKEN")
    if admin:
        os.environ["SABYINYO_ADMIN_TOKEN"] = admin

    out_dir = default_out_dir(platform)
    print(f"[launch] output dir = {out_dir}")
    subprocess.run(
        [sys.executable, "-m", "training.finetune", "--config", config,
         "--out-dir", out_dir],
        check=True, env={**os.environ, "PYTHONPATH": "."},
    )


if __name__ == "__main__":
    launch()
