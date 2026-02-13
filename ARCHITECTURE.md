# System Architecture

## 1. Data Pipeline (The Transpiler)
We need to create a massive dataset to teach LLMs how to "speak" WASM.

- **Source:** High-quality C repositories from GitHub with doxygen documentation.
- **Process:**
    1. Clone repository (shallow, depth=1)
    2. Extract C functions with doc comments
    3. Compile to WASM using LLVM `clang --target=wasm32`
    4. Convert to WAT using `wasm-tools print`
    5. Map functions to their docstrings
- **Output:** JSONL format `{"instruction": "Calculate hamming distance", "output": "(module (func $hamming ...))"}`

## 2. The Runtime (`zrun`)
A secure wrapper around `Wasmtime`.

- **Language:** Rust
- **Features:**
    - JIT Compilation (Near-native speed)
    - Capabilities Security Model (Cap-based sandbox)
    - `wasi` support for I/O operations

## 3. The Protocol (`.zero`)
While compliant with WASM, the `.zero` format adds a header for:
- **Digital Signature:** For verification
- **Metadata:** AI-readable hints for optimization

## 4. Tech Stack

```
┌─────────────────────────────────────────────┐
│              DATA PIPELINE                  │
├─────────────────────────────────────────────┤
│  C Source → LLVM clang → WASM → WAT → JSONL │
│                                             │
│  Tools: clang, lld, wasm-tools, Python      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              MODEL TRAINING                 │
├─────────────────────────────────────────────┤
│  JSONL Dataset → Fine-tune LLM              │
│                                             │
│  Base: Llama-3-8B / Mistral-7B / CodeLlama  │
│  Method: LoRA / QLoRA                       │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              ZERO RUNTIME                   │
├─────────────────────────────────────────────┤
│  Prompt → Model → WAT → WASM → Execute      │
│                                             │
│  Runtime: Rust + wasmtime                   │
│  Security: Capability-based sandbox         │
└─────────────────────────────────────────────┘
```

## 5. Why C?

C was chosen over Rust for the data pipeline because:
1. **Cleaner WAT output:** No runtime overhead (panic handlers, etc.)
2. **Simpler compilation:** Direct clang → WASM without complex wrapper
3. **Smaller output size:** ~2-5KB vs ~1MB for Rust
4. **Better training data:** More focused, less noise
