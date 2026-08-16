"""
Prepares a text corpus into train.bin / val.bin of uint16 GPT-2 BPE token ids.

Two source options:
  1. "shakespeare" — same tiny corpus you started with (1MB). Good for a quick
     sanity check that the pipeline works, but too small to show a real quality jump.
  2. "gutenberg"    — a few hundred MB of public-domain books via HF `datasets`.
     This is the step that actually lets scale help. Swap in your own .txt file
     by pointing SOURCE_PATH at it instead.

Usage (in Colab):
    !python data_prep.py --source shakespeare
    !python data_prep.py --source gutenberg --limit_mb 300
"""
import argparse
import os
import numpy as np
import tiktoken
import urllib.request


def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = "input.txt"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_gutenberg(limit_mb=300):
    from datasets import load_dataset
    # streaming so we don't have to download the whole thing
    ds = load_dataset("sedthh/gutenberg_english", split="train", streaming=True)
    text_parts = []
    total_bytes = 0
    limit_bytes = limit_mb * 1024 * 1024
    for row in ds:
        t = row.get("TEXT") or row.get("text") or ""
        text_parts.append(t)
        total_bytes += len(t.encode("utf-8"))
        if total_bytes >= limit_bytes:
            break
    return "\n\n".join(text_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["shakespeare", "gutenberg"], default="shakespeare")
    ap.add_argument("--limit_mb", type=int, default=300, help="only used for gutenberg")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    if args.source == "shakespeare":
        text = load_shakespeare()
    else:
        text = load_gutenberg(args.limit_mb)

    print(f"corpus size: {len(text) / 1e6:.2f}M characters")

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)  # no special tokens injected
    print(f"encoded: {len(ids) / 1e6:.2f}M tokens, vocab_size={enc.n_vocab}")

    ids = np.array(ids, dtype=np.uint16)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.out_dir, "val.bin"))
    print(f"train.bin: {len(train_ids)} tokens, val.bin: {len(val_ids)} tokens")
    print(f"saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
