# ZeroLang Development Session Notes

## Overview
This document captures key decisions, results, and learnings from the Phase 1-2 development session.

---

## Phase 1: Data Pipeline

### Architecture Decision: C-Only Focus
- **Decision**: Removed Rust support, focusing entirely on C
- **Rationale**: 
  - C produces cleaner, smaller WAT output (~1.8KB avg vs larger Rust output)
  - C functions are more self-contained, fewer trait dependencies
  - Simpler compilation pipeline (no cargo, just clang)

### Pipeline Components
1. **CFunctionExtractor**: Extracts functions from C source files
2. **CWatCompiler**: Compiles C → WASM → WAT using LLVM clang + lld
3. **DatasetGenerator**: Orchestrates cloning, extraction, compilation

### Key Enhancement: Signature-as-Instruction
- **Problem**: Only ~186 pairs collected when requiring doc comments
- **Solution**: Use function signature as instruction when no doc comment exists
  ```
  Before: "Calculate factorial" → WAT (only with doc)
  After:  "Implement: int factorial(int n)" → WAT (always)
  ```
- **Result**: ~3x more training data (544 pairs)

### Compilation Requirements
- LLVM clang with WASM target: `/opt/homebrew/opt/llvm/bin/clang`
- LLD linker: `brew install llvm lld`
- WASM tools: `cargo install wasm-tools`

### C Wrapper Template
Added standard typedefs for standalone compilation:
```c
typedef signed char int8_t;
typedef unsigned char uint8_t;
// ... etc
#define NULL ((void*)0)
```

---

## Phase 2: Model Training (PoC)

### Training Setup
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct (small for PoC)
- **Method**: LoRA fine-tuning (r=16, alpha=32)
- **Format**: ChatML with system prompt

### Results Comparison

| Metric | v1 (186 samples) | v2 (544 samples) |
|--------|------------------|------------------|
| Train samples | 168 | 490 |
| Val samples | 18 | 54 |
| Epochs | 3 | 5 |
| Training loss | 0.65 → 0.31 | 0.23 |
| **Eval loss** | 0.30 | **0.09** |
| Training time | ~3 min | ~13 min |

### Key Finding
- Model learned WAT syntax structure (module, exports, globals)
- Eval loss improved significantly (70% reduction)
- **Limitation**: Still produces template output, not specific function logic
- **Root cause**: 544 samples insufficient for true generalization

---

## Hardware Recommendations

### Data Collection
- **Recommended**: Mac or any machine (CPU-bound)
- Git cloning and text processing don't benefit from GPU

### Model Training
| Hardware | Capability |
|----------|------------|
| Mac (MPS) | 0.5B model, batch=1-2, memory issues |
| RTX 5070 (12GB) | 7B-14B models, batch=8-16, stable |

**Recommendation**: Use GPU PC (RTX 5070) for training larger models.

---

## Next Steps (Priority Order)

### 1. Scale Data Collection (Critical)
- Target: 5,000-10,000 training pairs
- Add more small, self-contained C repositories
- Focus on algorithmic code (math, crypto, data structures)

### 2. Use Larger Model
- Qwen2.5-7B-Instruct or Llama-3.2-8B
- Requires GPU with 12GB+ VRAM
- Expected: Much better function logic generation

### 3. Optimize Training
- Gradient checkpointing for memory efficiency
- Increase max_length to 1024-2048
- More epochs (10-20)

---

## Repository Statistics

### Processed Repositories (Top Contributors)
| Repository | Pairs |
|------------|-------|
| TheAlgorithms/C | 151 |
| cesanta/mongoose | 110 |
| nothings/stb | 67 |
| jedisct1/libsodium | 48 |
| ctz/cifra | 41 |
| gingerBill/gb | 22 |

### Common Compilation Failures
1. **Undeclared functions**: `strlen`, `malloc`, `printf` - need stdlib
2. **Custom types**: `aes_block_t`, `state_t` - project-specific
3. **Macros**: `crypto_aead_*_KEYBYTES` - header dependencies

---

## File Structure

```
project-zero/
├── pipeline/
│   ├── generator.py      # Main data pipeline
│   ├── postprocess.py    # Merge, split, stats
│   └── repos.txt         # Curated C repository list
├── training/
│   ├── prepare_data.py   # Convert to ChatML format
│   ├── train.py          # LoRA fine-tuning script
│   ├── inference.py      # Model testing
│   └── requirements.txt  # Python dependencies
├── data/
│   ├── expanded_training.jsonl  # 544 raw pairs
│   ├── train_chatml_v2.jsonl    # 490 training (ChatML)
│   └── val_chatml_v2.jsonl      # 54 validation (ChatML)
└── models/
    ├── zerolang-poc/     # v1 model (186 samples)
    └── zerolang-v2/      # v2 model (544 samples)
```

---

## Commands Reference

### Data Collection
```bash
# Run pipeline with signature fallback (default)
python pipeline/generator.py -l pipeline/repos.txt -o data/output.jsonl

# Append to existing data
python pipeline/generator.py -l repos.txt -o data/output.jsonl --append

# Only functions with doc comments
python pipeline/generator.py -l repos.txt -o data/output.jsonl --require-doc
```

### Training
```bash
# Prepare data
python training/prepare_data.py data/train.jsonl -o data/train_chatml.jsonl -f chatml

# Train model (CPU/MPS)
python training/train.py --epochs 5 --batch-size 1 --max-length 512

# Train on GPU (recommended)
python training/train.py --epochs 10 --batch-size 8 --max-length 1024
```

### Inference
```bash
# Test model
python training/inference.py --model-path models/zerolang-v2 \
  --base-model "Qwen/Qwen2.5-0.5B-Instruct" \
  --instruction "Implement: int add(int a, int b)"

# Interactive mode
python training/inference.py --model-path models/zerolang-v2 --interactive
```

---

## Lessons Learned

1. **Data quality > quantity initially**: Well-documented functions produce better training signal
2. **Standalone compilation is hard**: Most real-world C code has dependencies
3. **Small models learn syntax, not semantics**: Need 7B+ for actual code generation
4. **MPS has memory limits**: Use CPU or CUDA for stability with larger models
5. **Signature format works**: "Implement: void foo(int x)" is effective instruction format

---

*Last updated: February 2026*
