"""
Path B: fine-tune an existing small PRETRAINED model with LoRA.

Why this works better than Path A: TinyLlama-1.1B (or GPT-2) already has real
language competence and world knowledge from proper-scale pretraining (done by
someone else, with far more compute than a T4 has). Your job here is just the
alignment layer — the same conceptual step that turns GPT-3 into ChatGPT.

LoRA (Low-Rank Adaptation) freezes the base model and trains small injected
low-rank matrices instead of all 1.1B params — this is what makes it fit on
a T4 at all. Base model stays untouched on disk; you get a tiny adapter file.

Usage:
    !python finetune_pretrained.py --model tiny-llama --epochs 3
    !python finetune_pretrained.py --model gpt2 --epochs 3     # smaller/faster to try first
"""
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_IDS = {
    "gpt2": "gpt2",                                        # 124M — fastest to iterate on
    "gpt2-medium": "gpt2-medium",                           # 355M
    "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",     # 1.1B — best quality on a T4
}

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def format_example(tokenizer, row, max_len=512):
    instruction = row["instruction"]
    if row.get("context"):
        instruction += "\n\n" + row["context"]
    text = PROMPT_TEMPLATE.format(instruction=instruction, response=row["response"]) + tokenizer.eos_token

    tokenized = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")
    # mask the prompt portion out of the loss, same idea as Path A
    prompt_only = PROMPT_TEMPLATE.format(instruction=instruction, response="")
    prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=max_len)["input_ids"])

    labels = list(tokenized["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100  # HF convention: -100 is ignored by cross_entropy
    tokenized["labels"] = labels
    return tokenized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS.keys()), default="tiny-llama")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", default="checkpoints_lora")
    ap.add_argument("--limit", type=int, default=None, help="cap dataset size for quick runs")
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                 # rank of the low-rank update — 8-32 is typical for this scale
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] if "llama" in model_id.lower() else ["c_attn"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()   # sanity check: should be <1% of total params

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    ds = ds.map(lambda row: format_example(tokenizer, row), remove_columns=ds.column_names)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(f"{args.out_dir}/final_adapter")
    tokenizer.save_pretrained(f"{args.out_dir}/final_adapter")
    print(f"saved LoRA adapter to {args.out_dir}/final_adapter")


if __name__ == "__main__":
    main()
