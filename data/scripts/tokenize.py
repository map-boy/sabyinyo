import json

from tokenizers import ByteLevelBPETokenizer


def tokenize_file(tokenizer_path, input_path, output_path):
    tokenizer = ByteLevelBPETokenizer(
        f"{tokenizer_path}/vocab.json", f"{tokenizer_path}/merges.txt"
    )
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            ids = tokenizer.encode(line.strip()).ids
            f_out.write(json.dumps({"input_ids": ids}) + "\n")


if __name__ == "__main__":
    tokenize_file("tokenizer", "data/processed/train.txt", "data/processed/train.jsonl")
