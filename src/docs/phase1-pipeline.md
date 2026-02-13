# Phase 1: The Great Transpilation - Pipeline Documentation

## Overview
The data pipeline converts C source code from GitHub repositories into WAT (WebAssembly Text Format) training pairs for the ZeroLang model.

## Pipeline Flow
```
GitHub Repo → Clone → Extract C Functions → Filter (documented + simple) → Compile to WASM → Convert to WAT → JSONL
```

## Components

### 1. GitHubCloner
- Shallow clones repos (`--depth=1`) for efficiency
- Manages temporary directories
- Auto-cleanup on completion

### 2. CFunctionExtractor
- Walks `.c` and `.h` files, skips `test/`, `build/`, `.git/`
- Extracts functions with doxygen comments (`/** */` or `///`)
- Filters out complex functions (asm, volatile, >50 lines)

### 3. CWatCompiler
- Wraps functions with standard type definitions (uint8_t, etc.)
- Uses LLVM `clang --target=wasm32` with `lld` linker
- Converts `.wasm` → `.wat` via `wasm-tools print`

### 4. DatasetGenerator
- Orchestrates the full pipeline
- Hash-based deduplication
- Outputs JSONL: `{"instruction": "...", "output": "...", "metadata": {...}}`

## Usage
```bash
# Single repo
python3 pipeline/generator.py -r https://github.com/user/repo -o data/output.jsonl

# Multiple repos
python3 pipeline/generator.py -r https://github.com/a/b -r https://github.com/c/d

# From file
python3 pipeline/generator.py -l pipeline/repos.txt -o data/output.jsonl

# Verbose mode (show compilation errors)
python3 pipeline/generator.py -l pipeline/repos.txt -o data/output.jsonl -v
```

## Dependencies
- LLVM `clang` with WASM target (`/opt/homebrew/opt/llvm/bin/clang`)
- `lld` linker
- `wasm-tools`
- `git`

## Install Dependencies (macOS)
```bash
brew install llvm lld
cargo install wasm-tools
```

## Output Format (JSONL)
```json
{
  "instruction": "Function to calculate the Hamming distance between two strings",
  "output": "(module\n  (func $hamming_distance ...)\n  ...\n)",
  "metadata": {
    "function_name": "hamming_distance",
    "source_file": "src/strings/hamming.c",
    "repo_url": "https://github.com/...",
    "signature": "int hamming_distance(const char* s1, const char* s2)"
  }
}
```

## C Wrapper Template
The compiler adds standard type definitions for standalone compilation:
- `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, etc.
- `size_t`, `BYTE`, `WORD`, `DWORD`
- `NULL` definition

## Limitations
- Only simple, standalone functions (no external dependencies)
- Functions must have doxygen-style doc comments
- No inline assembly, volatile, or goto
- Max 50 lines per function

## Post-processing
```bash
# Merge multiple JSONL files
python3 pipeline/postprocess.py merge file1.jsonl file2.jsonl -o combined.jsonl

# Create train/validation split
python3 pipeline/postprocess.py split combined.jsonl --train train.jsonl --val val.jsonl

# View statistics
python3 pipeline/postprocess.py stats combined.jsonl
```
