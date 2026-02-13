# ZeroLang: The Complete Roadmap

> **Vision:** Eliminate the "Human Syntax Tax" — AI generates optimized bytecode directly, skipping legacy text-based programming languages entirely.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE ZEROLANG STACK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LAYER 4: APPLICATIONS                                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│   │  AI Agents  │  │  Serverless │  │    Edge     │  │   Browser   │       │
│   │  (Autonomous│  │  Functions  │  │  Computing  │  │    Apps     │       │
│   │   Code Gen) │  │             │  │             │  │             │       │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│          │                │                │                │               │
│   ───────┴────────────────┴────────────────┴────────────────┴───────────   │
│                                                                             │
│   LAYER 3: ZEROLANG MODEL (The Brain)                                       │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Fine-tuned LLM: "Instruction → Optimized WASM/IR"              │       │
│   │  • 7B-8B parameters, quantized for edge deployment              │       │
│   │  • Trained on millions of [docstring → bytecode] pairs          │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                    │                                        │
│   ─────────────────────────────────┴────────────────────────────────────   │
│                                                                             │
│   LAYER 2: ZERO RUNTIME (zrun)                                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Rust-based WASM executor with:                                 │       │
│   │  • JIT compilation (wasmtime)                                   │       │
│   │  • Capability-based security sandbox                            │       │
│   │  • WASI support for I/O                                         │       │
│   │  • Hot-reload & live patching                                   │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                    │                                        │
│   ─────────────────────────────────┴────────────────────────────────────   │
│                                                                             │
│   LAYER 1: THE .ZERO FORMAT                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  WASM-compatible binary with ZeroLang header:                   │       │
│   │  • Digital signature for verification                           │       │
│   │  • AI-readable optimization hints                               │       │
│   │  • Dependency graph metadata                                    │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: The Great Transpilation (Data Pipeline)

**Goal:** Build the world's largest [Human Intent → WASM] dataset from C code.

### 1.1 Infrastructure Setup
- [x] Define project architecture (`ARCHITECTURE.md`)
- [x] Create pipeline script structure (`pipeline/generator.py`)
- [x] Install toolchain dependencies
  - [x] LLVM `clang` with WASM backend ✅
  - [x] `lld` linker for WASM ✅
  - [x] `wasm-tools` for WASM↔WAT conversion ✅

### 1.2 C Pipeline
- [x] Implement C function extractor with doxygen comment parsing ✅
- [x] Add LLVM `clang` → WASM compilation path ✅
- [x] Create curated C repository list (`repos.txt` - 35+ repos)
- [x] Test pipeline (TheAlgorithms/C → 67 pairs) ✅
- [x] Implement deduplication (hash-based) ✅
- [x] Create validation split tool (`postprocess.py`) ✅
- [ ] Run pipeline at scale
- [ ] Target: **50,000+ training pairs**

### 1.3 Data Quality
- [x] Hash-based deduplication ✅
- [x] Train/validation split (10%) ✅
- [ ] Instruction paraphrasing (GPT-4 augmentation)
- [ ] WAT validation checks

### 1.4 Deliverables
```
data/
├── training.jsonl           # C → WAT pairs
├── validation.jsonl         # 10% validation set
└── metadata.json            # Dataset statistics
```

---

## Phase 2: Model Training (The Brain)

**Goal:** Fine-tune an LLM to generate valid, optimized WASM from natural language.

### 2.1 Base Model Selection
- [ ] Evaluate candidates:
  - [ ] **Llama-3-8B** (Meta, permissive license)
  - [ ] **Mistral-7B** (Fast, efficient)
  - [ ] **CodeLlama-7B** (Code-specialized)
  - [ ] **DeepSeek-Coder-7B** (Strong code understanding)
- [ ] Benchmark on code generation tasks
- [ ] Select final base model

### 2.2 Training Infrastructure
- [ ] Set up training environment
  - [ ] GPU cluster (A100 x 4 recommended)
  - [ ] Or cloud: Lambda Labs / RunPod / Modal
- [ ] Install training stack:
  - [ ] `transformers` + `accelerate`
  - [ ] `peft` (LoRA/QLoRA)
  - [ ] `bitsandbytes` (quantization)
  - [ ] `wandb` (experiment tracking)

### 2.3 Fine-tuning Strategy
- [ ] **Stage 1: Instruction Tuning**
  - [ ] Format: `<instruction>{docstring}</instruction><output>{wat}</output>`
  - [ ] Full fine-tune or LoRA (rank 64-128)
  - [ ] 3-5 epochs on full dataset
  
- [ ] **Stage 2: RLHF / DPO (Optional)**
  - [ ] Generate multiple WAT outputs per instruction
  - [ ] Rank by: compilation success, code size, execution speed
  - [ ] Train reward model or use DPO

### 2.4 Evaluation & Benchmarks
- [ ] Create ZeroBench evaluation suite:
  - [ ] **Compilation Rate:** % of outputs that compile
  - [ ] **Functional Correctness:** % that pass test cases
  - [ ] **Code Efficiency:** Binary size, execution time
- [ ] Compare against baselines (GPT-4 + compiler, Claude + compiler)

### 2.5 Deliverables
```
models/
├── zerolang-v0.1/           # Initial fine-tuned model
├── zerolang-v0.1-gguf/      # Quantized for deployment
└── training_logs/           # W&B artifacts
```

---

## Phase 3: Zero Runtime (zrun)

**Goal:** Build a production-ready WASM execution environment.

### 3.1 Core Runtime
- [ ] Initialize Rust project: `cargo new zrun`
- [ ] Integrate `wasmtime` as execution engine
- [ ] Implement `.zero` format parser
- [ ] Basic CLI: `zrun execute <file.zero>`

### 3.2 Security & Sandboxing
- [ ] Implement capability-based permissions
- [ ] Resource limits (memory, CPU time, fuel)
- [ ] Syscall filtering

### 3.3 WASI Integration
- [ ] Implement WASI preview interfaces
- [ ] Virtual filesystem support

### 3.4 Deliverables
```
zrun/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── runtime/             # Execution engine
│   └── sandbox/             # Security layer
└── Cargo.toml
```

---

## Phase 4-6: Future Phases

### Phase 4: The Decompiler (zread)
- WAT → Pseudo-code converter
- AI-powered explanation

### Phase 5: Ecosystem & Tooling
- VSCode extension
- Package registry
- Cloud deployment

### Phase 6: Advanced Capabilities
- Multi-module generation
- Self-improvement loop
- Formal verification

---

## Current Status

```
Phase 1: The Great Transpilation ✅
├── [x] Architecture defined
├── [x] Pipeline script (C only, clean WAT output)
├── [x] Toolchain setup (LLVM clang, lld, wasm-tools)
├── [x] Data collection (186 pairs, ~1.8KB avg WAT size)
└── [x] Train/val split (168/18)

Phase 2: Model Training (PoC)  ← YOU ARE HERE
├── [x] Training scripts created
├── [x] Data converted to ChatML format
├── [ ] Install dependencies
├── [ ] Run training
└── [ ] Test inference
```

**Next Action:** Install training dependencies and run PoC training.

### Quick Commands
```bash
# Run pipeline (full scale)
python3 pipeline/generator.py -l pipeline/repos.txt -o data/training.jsonl

# Merge and deduplicate (if multiple runs)
python3 pipeline/postprocess.py merge data/*.jsonl -o data/combined.jsonl

# Create train/val split
python3 pipeline/postprocess.py split data/combined.jsonl --train data/train.jsonl --val data/val.jsonl

# Check statistics
python3 pipeline/postprocess.py stats data/training.jsonl
```

---

## Success Metrics

### Phase 1 (Data)
- [ ] 50,000+ high-quality training pairs
- [ ] <5% duplicate rate
- [ ] 100% valid WAT output
- [ ] ~2-5KB average WAT size

### Phase 2 (Model)
- [ ] 85%+ compilation success rate
- [ ] 70%+ functional correctness on benchmarks
- [ ] <2s generation latency (quantized)

### Phase 3 (Runtime)
- [ ] <10ms cold start time
- [ ] Memory-safe execution (0 CVEs)
- [ ] WASI compliance

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Source Language | C (well-documented repos) |
| Compilation | LLVM clang → WASM |
| WAT Conversion | wasm-tools |
| Dataset Format | JSONL |
| Model | Llama-3 / Mistral / CodeLlama |
| Training | transformers + peft |
| Runtime | Rust + wasmtime |

---

*"The best code is no code. The second best is code humans never have to read."*
