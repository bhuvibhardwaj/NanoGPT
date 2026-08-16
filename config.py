from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257     # GPT-2 BPE vocab (tiktoken "gpt2")
    block_size: int = 256       # context length
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 384           # must be divisible by n_head
    dropout: float = 0.1
    bias: bool = False          # no bias in Linear/LayerNorm -> small speed/quality win


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints"
    data_dir: str = "data"

    batch_size: int = 64
    grad_accum_steps: int = 1          # effective batch = batch_size * grad_accum_steps
    max_iters: int = 6000
    eval_interval: int = 250
    eval_iters: int = 100

    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    lr_decay_iters: int = 6000         # usually == max_iters
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    device: str = "cuda"
    dtype: str = "fp16"                # T4 = Turing, no bf16 tensor cores -> use fp16. On A100/newer, switch to "bf16".
    compile_model: bool = False        # torch.compile gives little/no speedup on T4; enable on newer GPUs
