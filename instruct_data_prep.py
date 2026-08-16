"""
Builds an instruction-tuning dataset for your OWN from-scratch GPT (Path A).

Format per example, using GPT-2 BPE + a few special separator strings
(no new vocab needed — we just reuse plain text markers the model can learn):

    <|user|>
    {instruction}
    <|assistant|>
    {response}<|end|>

We tokenize the whole thing but store a per-token LOSS MASK: 0 over the
<|user|> block, 1 over the <|assistant|> response. This is what makes it
"instruction tuning" rather than "more pretraining" — the model is only ever
penalized for getting the RESPONSE wrong, so it learns to produce answers
conditioned on the prompt rather than learning to continue prompts.

Usage:
    !python instruct_data_prep.py --limit 20000
"""
import argparse
import numpy as np
import tiktoken

USER_TAG = "\n<|user|>\n"
ASSISTANT_TAG = "\n<|assistant|>\n"
END_TAG = "<|end|>"


def build_example(enc, instruction, response):
    prompt_ids = enc.encode_ordinary(USER_TAG + instruction + ASSISTANT_TAG)
    response_ids = enc.encode_ordinary(response + END_TAG)
    ids = prompt_ids + response_ids
    mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20000, help="max number of examples")
    ap.add_argument("--out_dir", default="data_instruct")
    args = ap.parse_args()

    from datasets import load_dataset
    # Dolly-15k: small, clean, human-written instruction/response pairs.
    # Swap for any dataset with instruction/response-shaped fields.
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    enc = tiktoken.get_encoding("gpt2")
    all_ids, all_mask = [], []

    n = min(args.limit, len(ds))
    for i in range(n):
        row = ds[i]
        instruction = row["instruction"]
        if row.get("context"):
            instruction = instruction + "\n\n" + row["context"]
        response = row["response"]
        ids, mask = build_example(enc, instruction, response)
        all_ids.extend(ids)
        all_mask.extend(mask)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    np.array(all_ids, dtype=np.uint16).tofile(f"{args.out_dir}/tokens.bin")
    np.array(all_mask, dtype=np.uint8).tofile(f"{args.out_dir}/mask.bin")
    print(f"wrote {len(all_ids)} tokens from {n} examples to {args.out_dir}/")


if __name__ == "__main__":
    main()
