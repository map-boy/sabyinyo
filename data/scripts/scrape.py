import os

from datasets import load_dataset


def pull_stack_subset(langs, out_dir, max_files_per_lang=50000):
    for lang in langs:
        ds = load_dataset(
            "bigcode/the-stack-v2-dedup",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True,
        )
        os.makedirs(f"{out_dir}/{lang}", exist_ok=True)
        for i, row in enumerate(ds):
            if i >= max_files_per_lang:
                break
            path = f"{out_dir}/{lang}/{i}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(row["content"])


if __name__ == "__main__":
    pull_stack_subset(["python", "typescript", "shell"], "data/raw")
