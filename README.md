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

## Current Status ✅

| Phase | Status | Description |
|-------|--------|-------------|
| Data Collection | ✅ Complete | 1000+ C→WAT pairs from 48 repos |
| Model Training | ✅ Complete | Qwen2.5-Coder-14B fine-tuned on H100 |
| API Deployment | ✅ Live | Gradio API on Colab |
| CLI Runtime | ✅ Working | End-to-end execution via `zerolang_cli.py` |

## Quick Start

### 1. Install Dependencies
```bash
# Clone the repo
git clone https://github.com/user/project-zero
cd project-zero

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install wasmtime gradio_client

# Install wasm-tools (for WAT→WASM conversion)
# macOS:
brew install wasm-tools

# Or download from: https://github.com/bytecodealliance/wasm-tools/releases
```

### 2. Run the CLI
```bash
# Start interactive CLI (replace with your Gradio URL)
python zerolang_cli.py --api https://YOUR-GRADIO-URL.gradio.live
```

### 3. Generate and Execute
```
zerolang> gen Implement: int add(int a, int b)
[✓] Generated 30 lines of WAT

zerolang> run add(5, 3)
╔════════════════════════════════╗
║  Result:                    8  ║
╚════════════════════════════════╝

zerolang> genrun Implement: int max(int a, int b) | max(10, 25)
[✓] Result: 25
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZeroLang Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌───────┐ │
│   │ Natural  │───▶│ Fine-tuned  │───▶│   WAT    │───▶│ WASM  │ │
│   │ Language │    │   LLM       │    │  Code    │    │ Binary│ │
│   └──────────┘    └─────────────┘    └──────────┘    └───────┘ │
│                                                          │      │
│                                          ┌───────────────▼────┐ │
│                                          │   wasmtime        │ │
│                                          │   (Execution)     │ │
│                                          └────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure
```
project-zero/
├── zerolang_cli.py       # 🎯 Main CLI - Start here!
├── pipeline/
│   ├── generator.py      # C → WAT data pipeline
│   ├── collect_large.py  # Large-scale data collection
│   └── postprocess.py    # Merge, dedup, split
├── training/
│   ├── train_cloud.py    # Cloud training script
│   └── inference.py      # Local inference test
├── zrun/
│   └── runtime.py        # WASM execution runtime
├── notebooks/
│   ├── Step1_Collect_Data.ipynb  # Colab data collection
│   ├── Step2_Train_Model.ipynb   # Colab training (H100)
│   └── Test_Environment.ipynb    # Environment verification
└── data/                 # Generated training data
```

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `gen <instruction>` | Generate WAT from instruction | `gen Implement: int add(int a, int b)` |
| `run <call>` | Execute last generated WAT | `run add(5, 3)` |
| `genrun <instr> \| <call>` | Generate and run in one step | `genrun Implement: int mul(int a, int b) \| mul(6, 7)` |
| `wat` | Show last generated WAT | |
| `clear` | Clear screen | |
| `help` | Show help | |
| `quit` | Exit | |

## Training Your Own Model

See [CLOUD_TRAINING.md](CLOUD_TRAINING.md) for detailed instructions on:
1. Collecting training data (free CPU on Colab)
2. Training the model (H100 GPU on Colab)
3. Deploying the API

## Tech Stack

| Component | Technology |
|-----------|------------|
| Training Data | C code from GitHub → WAT via LLVM |
| Model | Qwen2.5-Coder-14B (LoRA fine-tuned) |
| API | Gradio (hosted on Colab) |
| Runtime | wasmtime (Python + wasm-tools) |
| CLI | Python (zerolang_cli.py) |

## Limitations

- Currently works best with simple mathematical functions
- Recursive functions may hit stack limits
- No I/O operations (console, file) - pure computation only

## Roadmap

- [x] Phase 1: Data Collection Pipeline
- [x] Phase 2: Model Fine-Tuning
- [x] Phase 3: CLI Runtime
- [ ] Phase 4: Larger dataset (10k+ examples)
- [ ] Phase 5: Support for more complex programs
- [ ] Phase 6: Local model deployment (quantized)

---

*"The best code is no code. The second best is code humans never have to read."*
