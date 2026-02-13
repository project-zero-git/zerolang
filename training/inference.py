#!/usr/bin/env python3
"""
ZeroLang Inference Script

Test the trained model by generating WAT from natural language instructions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: Path, base_model: str = None):
    """Load the fine-tuned model."""
    print(f"[INFO] Loading model from {model_path}...")
    
    # Check for GPU - use CPU for consistency with training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load as PEFT model first
    try:
        # Load base model
        if base_model is None:
            # Try to infer from adapter config
            import json
            config_path = model_path / "adapter_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
        
        if base_model:
            print(f"[INFO] Loading base model: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            model = PeftModel.from_pretrained(model, model_path)
        else:
            raise ValueError("Could not determine base model")
            
    except Exception as e:
        print(f"[INFO] Loading as full model (not PEFT): {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
    
    model.eval()
    return model, tokenizer, device


def generate_wat(
    model,
    tokenizer,
    device,
    instruction: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
):
    """Generate WAT code from an instruction."""
    
    system_prompt = (
        "You are ZeroLang, an AI that generates optimized WebAssembly (WAT) code. "
        "Given a natural language description of a function, output valid WAT code that implements it. "
        "Output only the WAT code, no explanations."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in generated:
        response = generated.split("<|assistant|>")[-1].strip()
    else:
        response = generated[len(prompt):].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Generate WAT with ZeroLang")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/zerolang-poc"),
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (optional, will try to infer)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Instruction to generate WAT for"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path, args.base_model)
    
    if args.interactive:
        print("\n" + "=" * 60)
        print("ZeroLang Interactive Mode")
        print("Type 'quit' to exit")
        print("=" * 60 + "\n")
        
        while True:
            try:
                instruction = input("Instruction: ").strip()
                if instruction.lower() in ['quit', 'exit', 'q']:
                    break
                if not instruction:
                    continue
                
                print("\nGenerating WAT...\n")
                wat = generate_wat(
                    model, tokenizer, device,
                    instruction,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print("Generated WAT:")
                print("-" * 40)
                print(wat)
                print("-" * 40 + "\n")
                
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    elif args.instruction:
        wat = generate_wat(
            model, tokenizer, device,
            args.instruction,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(wat)
    
    else:
        # Demo with sample instructions
        demo_instructions = [
            "Calculate the factorial of a number",
            "Add two integers and return the result",
            "Check if a number is prime",
        ]
        
        print("\n" + "=" * 60)
        print("ZeroLang Demo")
        print("=" * 60 + "\n")
        
        for instruction in demo_instructions:
            print(f"Instruction: {instruction}")
            print("-" * 40)
            wat = generate_wat(
                model, tokenizer, device,
                instruction,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(wat[:500] + "..." if len(wat) > 500 else wat)
            print("\n")


if __name__ == "__main__":
    main()
