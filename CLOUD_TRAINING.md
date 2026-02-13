# Cloud Training Guide

## Quick Start (Google Colab)

1. Open notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/project-zero-git/zerolang/blob/master/notebooks/ZeroLang_Training.ipynb)

2. Enable GPU: `Runtime → Change runtime type → A100`

3. Run all cells

## Manual Setup

### Option 1: Google Colab (Free/Pro)

```bash
# In Colab notebook
!git clone https://github.com/project-zero-git/zerolang.git
%cd zerolang
!pip install torch transformers datasets peft accelerate bitsandbytes
!python training/train_cloud.py --model qwen-7b --epochs 10
```

### Option 2: RunPod ($1-2/hour)

1. Create account at [runpod.io](https://runpod.io)
2. Launch GPU pod (A100 40GB recommended)
3. Run:

```bash
git clone https://github.com/project-zero-git/zerolang.git
cd zerolang
pip install torch transformers datasets peft accelerate bitsandbytes
python training/train_cloud.py --model qwen-7b --epochs 10 --batch-size 8
```

### Option 3: Vast.ai ($0.3-1/hour)

1. Create account at [vast.ai](https://vast.ai)
2. Rent RTX 4090 or A100
3. Same commands as RunPod

### Option 4: Lambda Labs ($1.1/hour)

1. Create account at [lambdalabs.com](https://lambdalabs.com)
2. Launch A100 instance
3. Same commands as RunPod

## Model Selection

### Recommended: Qwen2.5-Coder (Specialized for Code)

| GPU | VRAM | Recommended Model | Command |
|-----|------|-------------------|---------|
| T4 | 16GB | `qwen-coder-3b` | `--model qwen-coder-3b --batch-size 2` |
| RTX 4090 | 24GB | `qwen-coder-7b` | `--model qwen-coder-7b --batch-size 4` |
| A100 40GB | 40GB | `qwen-coder-7b` ⭐ | `--model qwen-coder-7b --batch-size 8` |
| A100 80GB | 80GB | `qwen-coder-14b` | `--model qwen-coder-14b --batch-size 4` |

### Why Qwen2.5-Coder?
- **5.5 trillion tokens** of code data in pretraining
- **HumanEval 70%+** - Best open-source code benchmark
- **Syntax-aware** - Understands code structure (critical for WAT)
- **LoRA fine-tuning proven** - Up to 9% improvement reported

## Training Commands

### Basic (defaults - uses qwen-coder-7b)
```bash
python training/train_cloud.py
```

### Recommended for A100
```bash
python training/train_cloud.py \
    --model qwen-coder-7b \
    --epochs 10 \
    --batch-size 8 \
    --max-length 1024
```

### Full options
```bash
python training/train_cloud.py \
    --model qwen-coder-7b \
    --epochs 10 \
    --batch-size 8 \
    --max-length 1024 \
    --lr 2e-4 \
    --lora-r 32 \
    --lora-alpha 64 \
    --output models/zerolang-coder-v1
```

### Alternative models
```bash
# DeepSeek-Coder (good alternative)
python training/train_cloud.py --model deepseek-coder-6.7b

# Smaller model for T4
python training/train_cloud.py --model qwen-coder-3b --batch-size 2
```

### With Weights & Biases logging
```bash
wandb login
python training/train_cloud.py --model qwen-7b --wandb
```

### For Llama models (requires HF token)
```bash
export HF_TOKEN="your_token_here"
python training/train_cloud.py --model llama-8b --hf-token $HF_TOKEN
```

## Expected Results

| Model | Dataset | Epochs | Training Time | Expected Eval Loss |
|-------|---------|--------|---------------|-------------------|
| qwen-coder-3b | 544 | 10 | ~30 min | ~0.05-0.07 |
| qwen-coder-7b | 544 | 10 | ~1 hour | ~0.02-0.04 |
| qwen-coder-7b | 5000+ | 10 | ~3-4 hours | ~0.01-0.02 |
| qwen-coder-14b | 5000+ | 10 | ~6-8 hours | ~0.008-0.015 |

## Cost Estimates

| Service | GPU | Hourly | Full Training (~2h) |
|---------|-----|--------|---------------------|
| Colab Pro | A100 | $10/month | ~$0 (included) |
| RunPod | A100 40GB | $1.5/hour | ~$3 |
| Vast.ai | RTX 4090 | $0.4/hour | ~$1 |
| Lambda | A100 | $1.1/hour | ~$2.5 |

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 1 or 2
- Reduce `--max-length` to 512
- Use smaller model (`qwen-3b` instead of `qwen-7b`)

### Slow Training
- Increase `--batch-size` if memory allows
- Check GPU utilization: `nvidia-smi -l 1`

### Model not found (Llama)
- Llama models are gated, need HF token
- Accept license at huggingface.co/meta-llama
- Pass `--hf-token YOUR_TOKEN`

## Download Trained Model

After training, download from Colab:
```python
from google.colab import files
!zip -r model.zip models/zerolang-*
files.download('model.zip')
```

Or from terminal:
```bash
scp -r user@cloud-ip:~/zerolang/models/zerolang-* ./
```
