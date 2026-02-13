#!/usr/bin/env python3
"""
ZeroLang Cloud Training Script

Optimized for cloud GPU environments (Colab, RunPod, Lambda, Vast.ai).
Supports larger models (7B+) with memory optimizations.

Usage:
    # Quick start (Colab)
    !python train_cloud.py --model qwen-7b --data ./data
    
    # Full training (RunPod/Lambda)
    python train_cloud.py --model llama-8b --epochs 10 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Supported models - Qwen2.5-Coder recommended for code generation
MODELS = {
    # Qwen2.5-Coder (RECOMMENDED - Best for code generation)
    "qwen-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",  # ⭐ Best balance
    "qwen-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    
    # Qwen2.5 General (fallback)
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct", 
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-14b": "Qwen/Qwen2.5-14B-Instruct",
    
    # DeepSeek-Coder (alternative)
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    
    # Llama (requires HF token)
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
}


def get_device_info():
    """Get device and memory info."""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = "cpu"
        gpu_memory = 0
        print("[CPU] No GPU detected")
    return device, gpu_memory


def get_quantization_config(gpu_memory: float, model_size: str):
    """Determine quantization based on GPU memory and model size."""
    # Model sizes in billions
    sizes = {"0.5b": 0.5, "1.5b": 1.5, "1b": 1, "3b": 3, "7b": 7, "8b": 8, "14b": 14}
    
    model_gb = sizes.get(model_size.split("-")[-1], 7)
    required_memory = model_gb * 2  # Rough estimate: 2GB per 1B params in fp16
    
    if gpu_memory >= required_memory * 1.5:
        print(f"[INFO] Using fp16 (enough memory)")
        return None, torch.float16
    elif gpu_memory >= required_memory * 0.5:
        print(f"[INFO] Using 4-bit quantization (limited memory)")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ), torch.float16
    else:
        print(f"[WARN] GPU memory very limited, using 4-bit with offload")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ), torch.float16


def process_example(example, tokenizer, max_length):
    """Process a single example."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="ZeroLang Cloud Training")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen-coder-7b",  # Best for code generation
        choices=list(MODELS.keys()),
        help="Model to fine-tune (recommended: qwen-coder-7b)"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("data"),
        help="Data directory containing train_chatml.jsonl and val_chatml.jsonl"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("models/zerolang-cloud"),
        help="Output directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (saves memory)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (for gated models like Llama)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ZeroLang Cloud Training")
    print("=" * 60)
    
    # Device info
    device, gpu_memory = get_device_info()
    
    # Model
    model_name = MODELS[args.model]
    print(f"[MODEL] {args.model} -> {model_name}")
    
    # Data paths
    train_file = args.data / "train_chatml_v2.jsonl"
    val_file = args.data / "val_chatml_v2.jsonl"
    
    if not train_file.exists():
        train_file = args.data / "train_chatml.jsonl"
        val_file = args.data / "val_chatml.jsonl"
    
    print(f"[DATA] Train: {train_file}")
    print(f"[DATA] Val: {val_file}")
    print(f"[OUTPUT] {args.output}")
    print()
    
    # HF Token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Quantization config
    quant_config, torch_dtype = get_quantization_config(gpu_memory, args.model)
    
    # Load tokenizer
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training if quantized
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # LoRA config
    print("[INFO] Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("[INFO] Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        }
    )
    
    print(f"[INFO] Train: {len(dataset['train'])} samples")
    print(f"[INFO] Val: {len(dataset['validation'])} samples")
    
    # Tokenize
    print("[INFO] Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: process_example(x, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),  # Effective batch ~16
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=device == "cuda",
        bf16=False,
        push_to_hub=False,
        report_to="wandb" if args.wandb else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    # Train
    print()
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    trainer.train()
    
    # Save
    print()
    print(f"[INFO] Saving model to {args.output}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)
    
    # Save training config
    config = {
        "base_model": model_name,
        "model_alias": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.lr,
    }
    with open(args.output / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
