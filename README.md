# Project Zero (ZeroLang)

**Human-readable code is a legacy format. It's time for AI-native execution.**

## The Vision
ZeroLang is an experimental project to eliminate the "Human Syntax Tax".
Instead of AI writing Python/JS that needs to be parsed and compiled, we are training AI to output optimized **WebAssembly (WASM)** directly.

**Equation:** `Prompt → LLM → Optimized Binary (.zero) → Execution`

## Why?
1. **Efficiency:** Skip the parsing/lexing/compiling overhead.
2. **Context:** Token-optimized format means 3x more logic in the same context window.
3. **Portability:** Runs on Edge, Cloud, and Browser via WASM.

## Architecture
* **The Format:** `.zero` (A superset of WASM/WAT).
* **The Model:** A fine-tuned LLM (7B-8B params) specialized in "Instruction-to-Bytecode".
* **The Runtime:** `zrun` - A lightweight Rust-based runner with sandboxing.
* **The Decompiler:** `zread` - An AI-powered reverse engineering tool for human verification.

## Project Structure
```
project-zero/
├── pipeline/
│   ├── generator.py      # C → WAT data pipeline
│   ├── postprocess.py    # Merge, dedup, split
│   └── repos.txt         # Curated C repository list
├── data/                 # Generated training data
├── src/docs/
│   ├── ROADMAP.md        # Detailed roadmap
│   └── phase1-pipeline.md
└── README.md
```

## Quick Start

### Prerequisites
```bash
# Install LLVM with WASM support
brew install llvm lld

# Install wasm-tools
cargo install wasm-tools
```

### Generate Training Data
```bash
# Run on a single repo
python3 pipeline/generator.py -r https://github.com/TheAlgorithms/C -o data/output.jsonl

# Run on curated list
python3 pipeline/generator.py -l pipeline/repos.txt -o data/training.jsonl
```

## Roadmap
- [x] **Phase 1: The Great Transpilation** (Building the Dataset: C → WASM)
- [ ] **Phase 2: Fine-Tuning** (Training the Model on the Dataset)
- [ ] **Phase 3: Zero Runtime** (Building the `zrun` CLI in Rust)

## Tech Stack
| Component | Technology |
|-----------|------------|
| Data Source | C (GitHub repositories) |
| Compilation | LLVM clang → WASM |
| Model | Llama-3 / Mistral |
| Runtime | Rust + wasmtime |

---

*"The best code is no code. The second best is code humans never have to read."*
