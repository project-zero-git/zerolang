# Modal Cloud Training Setup

## Overview
Modal.com üzerinde serverless GPU ile ZeroLang model fine-tuning.

## Neden Modal?
- **Pay-per-use**: Sadece eğitim süresi kadar ödeme (~$0.80/saat A10G)
- **Sıfır infra yönetimi**: GPU provisioning otomatik
- **Hızlı cold start**: ~30 saniye
- **HuggingFace cache**: Model indirmeleri cache'lenir

## GPU Seçenekleri

| GPU | VRAM | Fiyat/saat | Kullanım |
|-----|------|------------|----------|
| T4 | 16GB | ~$0.30 | Test/debug |
| A10G | 24GB | ~$0.80 | QLoRA 7B modeller |
| A100-40GB | 40GB | ~$2.50 | Full fine-tune |
| A100-80GB | 80GB | ~$3.50 | Büyük modeller |

## Kurulum

```bash
# Modal CLI kurulumu
pip install modal

# Auth (ilk seferde)
modal setup
```

## Kullanım

```bash
# Dataset'i Modal volume'a yükle
modal run training/modal_train.py::upload_data

# Eğitimi başlat
modal run training/modal_train.py::train

# Modeli indir
modal run training/modal_train.py::download_model
```

## Maliyet Tahmini

- **3B model (Qwen2.5-3B)**: ~1-2 saat = ~$1-2
- **7B model**: ~3-4 saat = ~$3-4
- **Dataset boyutu**: 10K sample için ~1 saat

## Dosya Yapısı

```
training/
├── train.py           # Local training (mevcut)
├── modal_train.py     # Modal cloud training
└── requirements.txt   # Dependencies
```
