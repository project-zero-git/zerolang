#!/usr/bin/env python3
"""
ZeroLang PoC Training Script

Fine-tunes a small language model on instruction-to-WAT pairs using LoRA.
Designed for quick proof-of-concept testing.

Requirements:
    pip install transformers datasets peft accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def format_chat_template(example, tokenizer):
    """Format example using ChatML template."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description="Train ZeroLang PoC model")
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/train_chatml.jsonl"),
        help="Training data file (ChatML format)"
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/val_chatml.jsonl"),
        help="Validation data file (ChatML format)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model for faster training
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/zerolang-poc"),
        help="Output directory for model"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,  # Reduced for memory efficiency
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ZeroLang PoC Training")
    print("=" * 60)
    print(f"Base model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_r}")
    print()
    
    # Check for GPU
    # MPS can have memory issues with large models, use CPU for stability
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  # Skip MPS due to memory limitations
    print(f"[INFO] Using device: {device}")
    
    # Load tokenizer
    print(f"[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    # Configure LoRA
    print(f"[INFO] Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"[INFO] Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.val_file),
        }
    )
    
    # Format and tokenize
    print(f"[INFO] Processing dataset...")
    
    def process_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(
        process_example,
        remove_columns=dataset["train"].column_names,
    )
    
    print(f"[INFO] Train samples: {len(tokenized_dataset['train'])}")
    print(f"[INFO] Val samples: {len(tokenized_dataset['validation'])}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Increased for smaller batch
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        fp16=device == "cuda",
        push_to_hub=False,
        report_to="none",
        dataloader_pin_memory=False,  # For MPS compatibility
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
    print(f"\n[INFO] Starting training...")
    trainer.train()
    
    # Save
    print(f"\n[INFO] Saving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
