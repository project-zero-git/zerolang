"""
ZeroLang Modal Cloud Training

Serverless GPU üzerinde QLoRA fine-tuning.

Kullanım:
    # İlk kurulum
    pip install modal
    modal setup

    # Dataset yükle
    modal run training/modal_train.py::upload_data

    # Eğitimi başlat
    modal run training/modal_train.py::train

    # Modeli indir
    modal run training/modal_train.py::download_model
"""

import modal

# Modal app tanımı
app = modal.App("zerolang-training")

# Persistent volume - model ve data için
volume = modal.Volume.from_name("zerolang-data", create_if_missing=True)
VOLUME_PATH = "/data"

# Container image - tüm dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "scipy",
    )
)

# Training config
CONFIG = {
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",  # Kod için optimize 7B model
    "max_length": 1024,
    "epochs": 3,  # 7B için 3 epoch yeterli
    "batch_size": 2,  # 7B için düşür
    "gradient_accumulation_steps": 16,  # Effective batch = 32
    "learning_rate": 1e-4,  # 7B için biraz düşük LR
    "lora_r": 64,  # Daha yüksek rank, daha iyi kalite
    "lora_alpha": 128,
    "use_4bit": True,
}


@app.function(volumes={VOLUME_PATH: volume})
def upload_data():
    """Local dataset'i Modal volume'a yükle."""
    import os
    from pathlib import Path

    # Local dosyaları oku ve volume'a yaz
    local_files = [
        "data/train_chatml_v2.jsonl",
        "data/val_chatml_v2.jsonl",
    ]

    volume_data_dir = Path(VOLUME_PATH) / "dataset"
    volume_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Volume path: {VOLUME_PATH}")
    print(f"[INFO] Dataset dir: {volume_data_dir}")

    # Bu fonksiyon modal run ile çalıştırılacak
    # Local dosyalar modal.Mount ile gönderilecek
    print("[INFO] Dataset upload için local mount kullanın:")
    print("  modal run --mount data:/data/local training/modal_train.py::upload_data")

    volume.commit()
    print("[SUCCESS] Volume hazır!")


@app.local_entrypoint()
def upload_local_data():
    """Local'den dataset yükle."""
    from pathlib import Path

    train_file = Path("data/train_chatml_v2.jsonl")
    val_file = Path("data/val_chatml_v2.jsonl")

    if not train_file.exists():
        print(f"[ERROR] {train_file} bulunamadı!")
        return

    print(f"[INFO] Uploading {train_file}...")
    with open(train_file, "rb") as f:
        train_data = f.read()

    print(f"[INFO] Uploading {val_file}...")
    with open(val_file, "rb") as f:
        val_data = f.read()

    # Remote'a gönder
    _upload_to_volume.remote(train_data, val_data)
    print("[SUCCESS] Dataset uploaded!")


@app.function(volumes={VOLUME_PATH: volume})
def _upload_to_volume(train_data: bytes, val_data: bytes):
    """Dataset'i volume'a yaz."""
    from pathlib import Path

    dataset_dir = Path(VOLUME_PATH) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "train.jsonl").write_bytes(train_data)
    (dataset_dir / "val.jsonl").write_bytes(val_data)

    volume.commit()
    print(f"[INFO] Saved to {dataset_dir}")


@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM - QLoRA için ideal
    timeout=3600 * 4,  # 4 saat max
    volumes={VOLUME_PATH: volume},
)
def train():
    """Modal GPU üzerinde training başlat."""
    import torch
    from pathlib import Path
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

    config = CONFIG
    dataset_dir = Path(VOLUME_PATH) / "dataset"
    output_dir = Path(VOLUME_PATH) / "models" / "zerolang-modal"

    print("=" * 60)
    print("ZeroLang Modal Training")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model: {config['model_name']}")
    print()

    # Tokenizer
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    bnb_config = None
    if config["use_4bit"]:
        print("[INFO] Configuring 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # A10G bf16 destekler
            bnb_4bit_use_double_quant=True,
        )

    # Model
    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if config["use_4bit"]:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    print("[INFO] Configuring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    print("[INFO] Loading dataset...")
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {train_file}\n"
            "Önce 'modal run training/modal_train.py' ile dataset yükleyin."
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        }
    )

    # Tokenize
    def process_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("[INFO] Processing dataset...")
    tokenized_dataset = dataset.map(
        process_example,
        remove_columns=dataset["train"].column_names,
    )

    print(f"[INFO] Train samples: {len(tokenized_dataset['train'])}")
    print(f"[INFO] Val samples: {len(tokenized_dataset['validation'])}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        bf16=True,  # A10G için
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train
    print("\n[INFO] Starting training...")
    trainer.train()

    # Save
    print(f"\n[INFO] Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Volume'u commit et
    volume.commit()

    print("\n" + "=" * 60)
    print("[SUCCESS] Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

    return str(output_dir)


@app.function(volumes={VOLUME_PATH: volume})
def download_model():
    """Eğitilmiş modeli listele."""
    from pathlib import Path

    model_dir = Path(VOLUME_PATH) / "models" / "zerolang-modal"

    if not model_dir.exists():
        print("[ERROR] Model bulunamadı. Önce training çalıştırın.")
        return

    print(f"[INFO] Model directory: {model_dir}")
    print("\nFiles:")
    for f in model_dir.iterdir():
        size = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size:.2f} MB")

    print("\n[INFO] Modeli indirmek için:")
    print(f"  modal volume get zerolang-data models/zerolang-modal ./models/")


@app.local_entrypoint()
def main():
    """Ana entrypoint - training başlat."""
    print("[INFO] Starting Modal training...")
    result = train.remote()
    print(f"[INFO] Training completed: {result}")
