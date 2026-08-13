from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data/processed/train.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<pad>", "<bos>", "<eos>", "<unk>",
        "<PRE>", "<SUF>", "<MID>",
        "<file_sep>", "<repo_name>",
        "<|python|>", "<|typescript|>", "<|bash|>", "<|english|>",
    ],
)

if __name__ == "__main__":
    tokenizer.save_model("tokenizer")
