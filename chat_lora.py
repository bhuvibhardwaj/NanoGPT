import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_IDS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_IDS.keys()), default="tiny-llama")
    ap.add_argument("--adapter", default="checkpoints_lora/final_adapter")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    model_id = MODEL_IDS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    prompt = PROMPT_TEMPLATE.format(instruction=args.instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text[len(prompt):].strip())


if __name__ == "__main__":
    main()
