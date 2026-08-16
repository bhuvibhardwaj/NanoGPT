import argparse
import torch
import tiktoken

from model import GPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weights_only=False: PyTorch 2.6+ defaults to True, which blocks loading the
    # GPTConfig object stored alongside the weights. Safe here since this is a
    # checkpoint you trained yourself, not a downloaded/third-party file.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint from iter {ckpt['iter']}, val_loss={ckpt['val_loss']:.4f}")

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode_ordinary(args.prompt)], dtype=torch.long, device=device)

    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
