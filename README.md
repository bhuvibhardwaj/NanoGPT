# NanoGPT - MiniGPT v2

Scaled-up rebuild of the char-level model in `bhuvibhardwaj/MiniGPT`. Same
attention math you already implemented, upgraded on the three axes that
actually move generation quality: tokenizer, data, and training procedure.

## What changed vs. your notebooks

| | v1 (your notebooks) | v2 (this) |
|---|---|---|
| Tokenizer | char-level, vocab=65 | GPT-2 BPE, vocab=50257 |
| Context | 64 tokens | 256 tokens |
| Layers / heads / dim | 2 / 4 / 64 | 8 / 8 / 384 |
| Dropout | none | 0.1 |
| LR schedule | flat 3e-4 | warmup + cosine decay |
| Weight decay | none | AdamW, decoupled (no decay on norms/bias) |
| Precision | fp32 | fp16 autocast (T4-friendly) |
| Attention | manual masked_fill + softmax | `F.scaled_dot_product_attention` (flash kernel) |
| Data | Tiny Shakespeare, 1MB | swap in Gutenberg subset, ~100-300MB |

Params at the default config: ~29M (GPT-2-small is 124M, so this sits
comfortably between your old model and a "real" GPT-2).

## Run in Colab (T4)

```bash
!git clone <your-repo-or-upload-these-files>
%cd minigpt
!pip install -r requirements.txt -q

# 1. prep data — start with shakespeare to confirm the pipeline works end to end
!python data_prep.py --source shakespeare

# once that runs clean, swap to the bigger corpus:
!python data_prep.py --source gutenberg --limit_mb 300

# 2. train
!python train.py

# 3. generate
!python generate.py --prompt "ROMEO:" --max_new_tokens 300
```

## Tuning for your GPU budget

Everything sizeable lives in `config.py`. If you hit OOM on the T4 (16GB):
- drop `batch_size` first (64 → 32 → 16)
- then `n_embd` (384 → 256)
- use `grad_accum_steps` to recover effective batch size without more memory

If training is stable and loss is still dropping at `max_iters`, that's your
signal to increase `max_iters` (and `lr_decay_iters` to match) rather than
touching architecture — undertraining is the most common reason a bigger
model looks worse than a smaller one.

## What to watch during training

`train.py` prints train/val loss every `eval_interval` steps. The gap between
them is your overfitting signal — if val loss stops improving while train
keeps dropping, that's dropout/data-size territory, not "add more layers"
territory.

---

## Making it chat-like

A base model (everything above) only continues text — it has no notion of
"answering a question." Getting chat-like behavior needs an instruction-tuning
(SFT) stage on top. Two versions, do either or both:

### Path A — SFT your own model (educational: you own the whole pipeline)

```bash
!python instruct_data_prep.py --limit 20000   # Dolly-15k instruction pairs
!python sft_own_model.py                      # loads checkpoints/best.pt, fine-tunes
!python chat_own_model.py --instruction "What is the capital of France?"
```

Ceiling to expect: it will learn the *shape* of Q&A (stop-tokens, response
formatting) but won't reliably know facts your ~29M-param, 300MB-corpus base
model was never big enough to absorb. Good for showing you understand SFT
end-to-end; not a source of truth.

### Path B — LoRA fine-tune a real pretrained model (actually useful results)

Starts from a properly-pretrained small model (TinyLlama-1.1B or GPT-2) and
only trains small LoRA adapter matrices on top — fits on a T4, keeps the base
frozen.

```bash
!python finetune_pretrained.py --model gpt2 --epochs 3          # try this first, fast
!python finetune_pretrained.py --model tiny-llama --epochs 3    # better quality, slower

!python chat_lora.py --model tiny-llama --instruction "Explain LoRA in one paragraph"
```

This is the closer analogue to what actually turns a base GPT into ChatGPT/
Claude: freeze a large pretrained model, train a small alignment layer on
instruction data. Real RLHF adds a further preference-learning stage on top
of this (reward model + PPO, or the simpler DPO) — worth knowing the term
exists, but SFT alone already gets you noticeably chat-like behavior.
