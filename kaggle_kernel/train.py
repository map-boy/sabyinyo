import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import subprocess
import sys

try:
    from tokenizers import ByteLevelBPETokenizer
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)
    from tokenizers import ByteLevelBPETokenizer

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "--force-reinstall",
    "torch==2.4.1",
    "--extra-index-url", "https://download.pytorch.org/whl/cu121"
], check=True)
import torch

try:
    from huggingface_hub import HfApi
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN_WRITE")
HF_REPO_ID = "map-boy/sabyinyo-codegen"
HF_ENABLED = HF_TOKEN is not None
if HF_ENABLED:
    try:
        _hf_api = HfApi(token=HF_TOKEN)
        _hf_api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    except Exception as _e:
        print(f"HF setup failed, disabling HF push: {_e}")
        HF_ENABLED = False
else:
    print("HF_TOKEN unavailable after retries - HF push disabled, checkpoints stay local")
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

if os.path.isdir("/content"):
    DATA_DIR = "/content/data"
    OUT_DIR = "/content/checkpoints"
elif os.path.isdir("/kaggle/input"):
    DATA_DIR = "/kaggle/input/datasets/mugishaalainpaisible/codegen-corpus-v1"
    OUT_DIR = "/kaggle/working/checkpoints"
else:
    DATA_DIR = os.environ.get("SABYINYO_DATA_DIR", "./data")
    OUT_DIR = os.environ.get("SABYINYO_OUT_DIR", "./checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)

# --- must match configs/model_config.yaml exactly ---
VOCAB_SIZE = 32000
DIM = 768
N_LAYERS = 12
N_HEADS = 12
SEQ_LEN = 2048

# --- must match configs/train_config.yaml exactly ---
BATCH_SIZE = 4
GRAD_ACCUM = 32
LR = 3e-4
WARMUP_STEPS = 2000
MAX_STEPS = 100000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
SAVE_EVERY = 2000
EVAL_EVERY = 1000
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95

SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>",
    "<PRE>", "<SUF>", "<MID>",
    "<file_sep>", "<repo_name>",
    "<|python|>", "<|typescript|>", "<|bash|>", "<|english|>",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============ model/layers.py (verbatim) ============
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, rope=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.rope = rope
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ============ model/architecture.py (verbatim) ============
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult=4):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads=max(1, n_heads // 4))
        self.norm1 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden=dim * ffn_mult)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CodeGenModel(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ============ model/checkpoint_utils.py (verbatim, out_dir repointed) ============
def save_checkpoint(model, optimizer, step, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step}.pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        path,
    )
    if HF_ENABLED:
        try:
            _hf_api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"checkpoints/step_{step}.pt",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print(f"pushed step_{step}.pt to {HF_REPO_ID}")
            _hf_api.upload_file(
                path_or_fileobj=path,
                path_in_repo="checkpoints/latest.pt",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            os.remove(path)
        except Exception as e:
            print(f"HF push failed for step_{step}.pt, keeping local copy: {e}")
    return path


# ============ training/pretrain.py (verbatim loss fn) ============
def cross_entropy(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


# ============ data pipeline ============
class TextChunkDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        n_chunks = (len(token_ids) - 1) // seq_len
        self.data = token_ids[: n_chunks * seq_len + 1]
        self.n_chunks = n_chunks

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]).to(device),
        "labels": torch.stack([b["labels"] for b in batch]).to(device),
    }


def load_tokenizer():
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    merges_path = os.path.join(DATA_DIR, "merges.txt")
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(
            f"tokenizer files not found under {DATA_DIR} -- check"
            f"kernel-metadata.json matches the dataset slug exactly."
        )
    tok = ByteLevelBPETokenizer(vocab_path, merges_path)
    return tok


def tokenize_corpus(tok, path, batch_lines=2000):
    token_ids = []
    buf = []
    total_lines = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= batch_lines:
                encodings = tok.encode_batch(buf)
                for e in encodings:
                    token_ids.extend(e.ids)
                total_lines += len(buf)
                buf = []
                if total_lines % 200000 == 0:
                    print(f"  tokenized {total_lines} lines, {len(token_ids)} tokens so far")
        if buf:
            encodings = tok.encode_batch(buf)
            for e in encodings:
                token_ids.extend(e.ids)
            total_lines += len(buf)
    print(f"  done: {total_lines} lines, {len(token_ids)} tokens total")
    return token_ids


def build_dataloaders():
    tok = load_tokenizer()
    corpus_path = os.path.join(DATA_DIR, "train.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"train.txt not found under {DATA_DIR}")

    print("Tokenizing corpus with real ByteLevelBPE tokenizer...")
    token_ids = tokenize_corpus(tok, corpus_path)

    if len(token_ids) < SEQ_LEN * 10:
        raise RuntimeError(
            f"Only {len(token_ids)} tokens produced -- check"
            f"Check that train.txt actually contains the corpus."
        )

    n_val = max(SEQ_LEN + 1, int(len(token_ids) * 0.01))
    train_ids = token_ids[:-n_val]
    val_ids = token_ids[-n_val:]

    train_ds = TextChunkDataset(train_ids, SEQ_LEN)
    val_ds = TextChunkDataset(val_ids, SEQ_LEN)

    print(f"train chunks: {len(train_ds)}, val chunks: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def evaluate(model, val_loader, max_batches=20):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            logits = model(batch["input_ids"])
            loss = cross_entropy(logits, batch["labels"])
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def main():
    if os.path.isdir("/kaggle/input"):
        print("DEBUG kaggle/input contents:", os.listdir("/kaggle/input"))
        for d in os.listdir("/kaggle/input"):
            print("DEBUG  ", d, "->", os.listdir(f"/kaggle/input/{d}"))
    print(f"device: {device}")
    print("DEBUG torch version:", torch.__version__)
    if device == "cuda":
        print("DEBUG torch cuda version:", torch.version.cuda)
        print("DEBUG gpu name:", torch.cuda.get_device_name(0))
        print("DEBUG gpu capability:", torch.cuda.get_device_capability(0))
    else:
        print("WARNING: no GPU detected, check runtime/accelerator settings.")

    train_loader, val_loader = build_dataloaders()

    model = CodeGenModel(VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, SEQ_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(ADAM_BETA1, ADAM_BETA2), weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    if HF_ENABLED:
        try:
            from huggingface_hub import hf_hub_download
            latest_path = hf_hub_download(repo_id=HF_REPO_ID, filename="checkpoints/latest.pt", token=HF_TOKEN)
            ckpt = torch.load(latest_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            print(f"resumed from step {start_step}")
        except Exception as e:
            print(f"no resumable checkpoint found, starting fresh: {e}")

    model.train()
    step = start_step
    optimizer.zero_grad()
    for epoch in range(1000):
        for batch in train_loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(batch["input_ids"])
                loss = cross_entropy(logits, batch["labels"]) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % EVAL_EVERY == 0:
                val_loss = evaluate(model, val_loader)
                print(f"[step {step}] train_loss={loss.item()*GRAD_ACCUM:.4f} val_loss={val_loss:.4f}")

            if step % SAVE_EVERY == 0 and step > 0:
                path = save_checkpoint(model, optimizer, step)
                print(f"saved checkpoint: {path}")

            step += 1
            if step >= MAX_STEPS:
                save_checkpoint(model, optimizer, step)
                print("training complete")
                return
        print(f"epoch {epoch} finished, continuing (step {step}/{MAX_STEPS})")


if __name__ == "__main__":
    main()

