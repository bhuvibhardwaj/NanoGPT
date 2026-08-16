"""
Supervised fine-tuning (SFT) for your own from-scratch GPT.

Loads the base checkpoint from train.py (checkpoints/best.pt) and continues
training on instruction data, but with the loss masked to only the response
tokens (see instruct_data_prep.py). LR is much lower than base pretraining —
this is a light nudge on top of already-learned representations, not
training from scratch.
"""
import os
import numpy as np
import torch
from torch.nn import functional as F

from config import GPTConfig
from model import GPT

BASE_CKPT = "checkpoints/best.pt"
DATA_DIR = "data_instruct"
OUT_DIR = "checkpoints_sft"

BLOCK_SIZE = 256
BATCH_SIZE = 32
SFT_LR = 3e-5          # ~10x lower than base pretraining LR — SFT should nudge, not overwrite
MAX_ITERS = 2000
EVAL_INTERVAL = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

tokens = np.memmap(f"{DATA_DIR}/tokens.bin", dtype=np.uint16, mode="r")
mask = np.memmap(f"{DATA_DIR}/mask.bin", dtype=np.uint8, mode="r")
assert len(tokens) == len(mask)


def get_batch():
    ix = torch.randint(len(tokens) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(tokens[i:i + BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64)) for i in ix])
    # mask aligned to the TARGET (y) position, since that's what the loss is over
    m = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + BLOCK_SIZE].astype(np.float32)) for i in ix])
    return x.to(device), y.to(device), m.to(device)


def masked_loss(logits, targets, mask):
    B, T, C = logits.shape
    loss_per_token = F.cross_entropy(
        logits.view(-1, C), targets.view(-1), reduction="none"
    ).view(B, T)
    mask = mask.clamp(min=1e-8)
    return (loss_per_token * mask).sum() / mask.sum()


def main():
    ckpt = torch.load(BASE_CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded base model, prior val_loss={ckpt['val_loss']:.4f}")

    optimizer = model.configure_optimizer(weight_decay=0.0, learning_rate=SFT_LR, betas=(0.9, 0.95))

    os.makedirs(OUT_DIR, exist_ok=True)
    model.train()
    for it in range(MAX_ITERS):
        x, y, m = get_batch()
        # model.forward's built-in loss is an unmasked mean over all positions,
        # which isn't what we want here — so we run the forward pass ourselves
        # and apply the mask before reducing.
        tok_emb = model.token_embedding(x)
        pos_emb = model.position_embedding(torch.arange(x.size(1), device=device))
        h = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            h = block(h)
        h = model.ln_f(h)
        logits_full = model.lm_head(h)
        loss = masked_loss(logits_full, y, m)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % EVAL_INTERVAL == 0:
            print(f"iter {it:5d} | sft loss {loss.item():.4f}")

    torch.save({"model": model.state_dict(), "config": cfg}, f"{OUT_DIR}/sft.pt")
    print(f"saved to {OUT_DIR}/sft.pt")


if __name__ == "__main__":
    main()
