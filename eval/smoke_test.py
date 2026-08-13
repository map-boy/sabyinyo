import torch

from model.architecture import CodeGenModel

m = CodeGenModel(vocab_size=100, dim=32, n_layers=2, n_heads=4, max_seq_len=16)
x = torch.randint(0, 100, (1, 8))
out = m(x)
assert out.shape == (1, 8, 100)
print("smoke test passed:", out.shape)
