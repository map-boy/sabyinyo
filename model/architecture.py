from torch import nn

from model.layers import GroupedQueryAttention, RMSNorm, SwiGLU


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
