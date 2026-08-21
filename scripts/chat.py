#!/usr/bin/env python3
"""Simple chatbot — talk to your trained ternary model.

Usage:
  .venv/bin/python scripts/chat.py --model bonsai-chat
  .venv/bin/python scripts/chat.py --model bonsai-chat --device cpu
  .venv/bin/python scripts/chat.py --model bonsai-chat --device cuda:0
"""
import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description="Chat with your ternary model")
    ap.add_argument("--model", default="bonsai-chat",
                    help="Path to trained model")
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:0, cuda:1, cpu")
    ap.add_argument("--system", default="You are a helpful assistant. Answer questions clearly and concisely.",
                    help="System prompt")
    ap.add_argument("--max-new", type=int, default=256,
                    help="Max tokens to generate per response")
    ap.add_argument("--temp", type=float, default=0.7,
                    help="Temperature (lower = more focused)")
    ap.add_argument("--no-thinking", action="store_true", default=True,
                    help="Disable Qwen3 thinking mode (default: off)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.0f}M parameters")
    print(f"System: {args.system}")
    print(f"\n{'='*60}")
    print(f"  Chat with your ternary model. Type 'quit' to exit.")
    print(f"{'='*60}\n")

    conversation = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        # Build prompt manually (no thinking mode)
        prompt = ""
        for msg in conversation:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tok(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                temperature=args.temp,
                do_sample=args.temp > 0,
                top_p=0.9,
                top_k=50,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tok.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up any thinking tokens if they slipped in
        for tag in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]:
            response = response.replace(tag, "")
        response = response.strip()

        print(f"Assistant: {response}\n")
        conversation.append({"role": "assistant", "content": response})

        # Keep conversation history manageable (last 10 turns + system)
        if len(conversation) > 21:
            conversation = [conversation[0]] + conversation[-20:]


if __name__ == "__main__":
    main()
