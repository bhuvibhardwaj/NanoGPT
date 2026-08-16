import argparse
import torch
import tiktoken

from model import GPT

USER_TAG = "\n<|user|>\n"
ASSISTANT_TAG = "\n<|assistant|>\n"
END_TAG = "<|end|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints_sft/sft.pt")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    prompt = USER_TAG + args.instruction + ASSISTANT_TAG
    idx = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)

    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    text = enc.decode(out[0].tolist())
    response = text[len(prompt):].split(END_TAG)[0]
    print(response.strip())


if __name__ == "__main__":
    main()
