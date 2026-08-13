import torch


def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.3):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    pass
