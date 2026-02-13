#!/usr/bin/env python3
"""
ZeroLang - Download and Test Model

This script helps you download and test the trained model from Google Drive or local zip.

Usage:
    # If model is in Google Drive
    python scripts/download_and_test.py --drive-path "/path/in/drive"
    
    # If you have a local zip
    python scripts/download_and_test.py --zip zerolang-model.zip
    
    # If model is already extracted
    python scripts/download_and_test.py --model-path models/zerolang-qwen-coder-14b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, output_dir: Path) -> Path:
    """Extract model from zip file."""
    print(f"[INFO] Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    
    # Find the model directory
    for item in output_dir.iterdir():
        if item.is_dir() and (item / "adapter_config.json").exists():
            return item
    
    # Check in models subdirectory
    models_dir = output_dir / "models"
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                return item
    
    raise FileNotFoundError("Could not find model in zip file")


def test_model(model_path: Path):
    """Test the model with sample prompts."""
    print(f"\n[INFO] Loading model from {model_path}...")
    
    # Import here to avoid slow startup
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"[INFO] Using device: {device}")
    
    # Load config to get base model
    config_path = model_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    base_model = config.get("base_model_name_or_path")
    
    print(f"[INFO] Base model: {base_model}")
    print(f"[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"[INFO] Loading base model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    if device not in ["cuda"]:
        model = model.to(device)
    
    print(f"[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print("\n" + "=" * 60)
    print("ZeroLang Model Ready!")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "Implement: int add(int a, int b)",
        "Implement: int max(int a, int b)", 
        "Implement: int factorial(int n)",
    ]
    
    for prompt in test_prompts:
        print(f"\n>>> {prompt}")
        print("-" * 40)
        
        messages = [
            {"role": "system", "content": "You are ZeroLang, an AI that generates WebAssembly (WAT) code."},
            {"role": "user", "content": prompt},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        # Truncate for display
        if len(response) > 400:
            response = response[:400] + "\n..."
        print(response)
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    
    return model, tokenizer


def interactive_mode(model, tokenizer, device):
    """Run interactive mode."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("Enter instruction: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue
            
            messages = [
                {"role": "system", "content": "You are ZeroLang, an AI that generates WebAssembly (WAT) code."},
                {"role": "user", "content": prompt},
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print("\n" + response + "\n")
            
        except KeyboardInterrupt:
            break
    
    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="Download and test ZeroLang model")
    parser.add_argument("--zip", type=Path, help="Path to model zip file")
    parser.add_argument("--model-path", type=Path, help="Path to extracted model directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive mode after test")
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    elif args.zip:
        model_path = extract_zip(args.zip, Path("models"))
    else:
        # Look for existing model
        default_paths = [
            Path("models/zerolang-qwen-coder-14b"),
            Path("models/zerolang-qwen-coder-14b-colab"),
            Path("models/zerolang-qwen-coder-7b"),
        ]
        model_path = None
        for p in default_paths:
            if p.exists() and (p / "adapter_config.json").exists():
                model_path = p
                break
        
        if model_path is None:
            print("ERROR: No model found. Please specify --zip or --model-path")
            sys.exit(1)
    
    # Verify model exists
    if not model_path.exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)
    
    if not (model_path / "adapter_config.json").exists():
        print(f"ERROR: Not a valid LoRA model directory: {model_path}")
        sys.exit(1)
    
    # Test model
    model, tokenizer = test_model(model_path)
    
    # Interactive mode
    if args.interactive:
        device = next(model.parameters()).device
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
