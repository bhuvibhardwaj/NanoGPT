import math
import os
import time
import numpy as np
import torch

from config import GPTConfig, TrainConfig
from model import GPT

mcfg = GPTConfig()
tcfg = TrainConfig()

os.makedirs(tcfg.out_dir, exist_ok=True)
device = tcfg.device if torch.cuda.is_available() else "cpu"
ptdtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[tcfg.dtype]
use_amp = device == "cuda" and tcfg.dtype in ("fp16", "bf16")
scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.dtype == "fp16"))


def get_batch(split):
    path = os.path.join(tcfg.data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - mcfg.block_size, (tcfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + mcfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + mcfg.block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it):
    # linear warmup then cosine decay to min_lr — this is the single biggest
    # training-stability upgrade over your flat 3e-4 for the whole run
    if it < tcfg.warmup_iters:
        return tcfg.learning_rate * (it + 1) / tcfg.warmup_iters
    if it > tcfg.lr_decay_iters:
        return tcfg.min_lr
    decay_ratio = (it - tcfg.warmup_iters) / (tcfg.lr_decay_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return tcfg.min_lr + coeff * (tcfg.learning_rate - tcfg.min_lr)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=ptdtype, enabled=use_amp):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    model = GPT(mcfg).to(device)
    print(f"model params: {model.num_params() / 1e6:.2f}M")

    if tcfg.compile_model:
        model = torch.compile(model)

    optimizer = model.configure_optimizer(
        weight_decay=tcfg.weight_decay, learning_rate=tcfg.learning_rate, betas=(0.9, 0.95)
    )

    best_val = float("inf")
    t0 = time.time()

    for it in range(tcfg.max_iters):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for micro in range(tcfg.grad_accum_steps):
            x, y = get_batch("train")
            with torch.autocast(device_type="cuda", dtype=ptdtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if it % tcfg.eval_interval == 0 or it == tcfg.max_iters - 1:
            losses = estimate_loss(model)
            dt = time.time() - t0
            print(f"iter {it:5d} | lr {lr:.2e} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {dt:.1f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": mcfg, "iter": it, "val_loss": best_val},
                    os.path.join(tcfg.out_dir, "best.pt"),
                )

    print(f"done. best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
