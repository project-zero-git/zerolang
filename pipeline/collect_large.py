#!/usr/bin/env python3
"""
ZeroLang Large-Scale Data Collection

Optimized pipeline for collecting 10,000+ high-quality C→WAT pairs.

Features:
- Parallel repository processing
- Quality filtering
- Deduplication
- Progress tracking
- Resume capability

Usage:
    # Full collection
    python pipeline/collect_large.py --target 10000 --output data/large_dataset.jsonl
    
    # Resume from checkpoint
    python pipeline/collect_large.py --resume --output data/large_dataset.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Set, Dict, Any, Optional
from datetime import datetime


# =============================================================================
# CURATED REPOSITORY LIST - Ordered by expected yield
# =============================================================================

REPOS = {
    # TIER 1: Algorithm Collections (High yield, high quality)
    "tier1_algorithms": [
        "https://github.com/TheAlgorithms/C",
        "https://github.com/fragglet/c-algorithms",
        "https://github.com/attractivechaos/klib",
        "https://github.com/srdja/Collections-C",
    ],
    
    # TIER 2: Cryptography (Good standalone functions)
    "tier2_crypto": [
        "https://github.com/B-Con/crypto-algorithms",
        "https://github.com/kokke/tiny-AES-c",
        "https://github.com/ctz/cifra",
        "https://github.com/983/SHA-256",
        "https://github.com/983/Num",
        "https://github.com/jedisct1/libsodium",  # Large but good
    ],
    
    # TIER 3: Data Structures
    "tier3_datastructures": [
        "https://github.com/troydhanson/uthash",
        "https://github.com/tezc/sc",
        "https://github.com/tidwall/hashmap.c",
        "https://github.com/tidwall/btree.c",
        "https://github.com/antirez/rax",
        "https://github.com/antirez/sds",
        "https://github.com/rxi/vec",
        "https://github.com/rxi/map",
        "https://github.com/clibs/buffer",
        "https://github.com/clibs/list",
    ],
    
    # TIER 4: String/Text Processing
    "tier4_string": [
        "https://github.com/sheredom/utf8.h",
        "https://github.com/jwerle/murmurhash.c",
        "https://github.com/skeeto/branchless-utf8",
        "https://github.com/antirez/linenoise",
    ],
    
    # TIER 5: Parsing/JSON
    "tier5_parsing": [
        "https://github.com/DaveGamble/cJSON",
        "https://github.com/zserge/jsmn",
        "https://github.com/kgabis/parson",
        "https://github.com/cesanta/frozen",
        "https://github.com/orangeduck/mpc",
    ],
    
    # TIER 6: Compression
    "tier6_compression": [
        "https://github.com/lz4/lz4",
        "https://github.com/richgel999/miniz",
        "https://github.com/ebiggers/libdeflate",
    ],
    
    # TIER 7: Math/Numerical
    "tier7_math": [
        "https://github.com/nothings/stb",
        "https://github.com/983/fft",
        "https://github.com/skeeto/hash-prospector",
        "https://github.com/lemire/clhash",
        "https://github.com/lemire/simdcomp",
        "https://github.com/lemire/SIMDxorshift",
    ],
    
    # TIER 8: Embedded/System
    "tier8_embedded": [
        "https://github.com/cesanta/mongoose",
        "https://github.com/nodejs/http-parser",
        "https://github.com/antirez/kilo",
        "https://github.com/jart/cosmopolitan",
    ],
    
    # TIER 9: Misc Utilities
    "tier9_misc": [
        "https://github.com/gingerBill/gb",
        "https://github.com/skeeto/optparse",
        "https://github.com/rxi/log.c",
        "https://github.com/sheredom/hashmap.h",
        "https://github.com/mackron/dr_libs",
        "https://github.com/ndevilla/iniparser",
        "https://github.com/benhoyt/inih",
    ],
}


# =============================================================================
# QUALITY FILTERS
# =============================================================================

@dataclass
class QualityConfig:
    """Configuration for quality filtering."""
    min_instruction_len: int = 10
    max_instruction_len: int = 500
    min_wat_lines: int = 3
    max_wat_lines: int = 300
    min_wat_chars: int = 50
    max_wat_chars: int = 15000
    require_func_export: bool = True
    banned_instructions: List[str] = field(default_factory=lambda: [
        "TODO", "FIXME", "XXX", "HACK", "test", "debug"
    ])


def passes_quality_filter(pair: Dict[str, Any], config: QualityConfig) -> bool:
    """Check if a training pair passes quality filters."""
    instruction = pair.get("instruction", "")
    output = pair.get("output", "")
    
    # Instruction length
    if len(instruction) < config.min_instruction_len:
        return False
    if len(instruction) > config.max_instruction_len:
        return False
    
    # WAT length
    wat_lines = output.count('\n') + 1
    if wat_lines < config.min_wat_lines:
        return False
    if wat_lines > config.max_wat_lines:
        return False
    
    if len(output) < config.min_wat_chars:
        return False
    if len(output) > config.max_wat_chars:
        return False
    
    # Must contain module and func
    if "(module" not in output:
        return False
    if "(func" not in output:
        return False
    
    # Check for banned words in instruction
    instruction_lower = instruction.lower()
    for banned in config.banned_instructions:
        if banned.lower() in instruction_lower:
            return False
    
    return True


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_instruction(instruction: str) -> List[str]:
    """Generate instruction variations for data augmentation."""
    variations = [instruction]  # Original
    
    # If it's an "Implement:" style instruction
    if instruction.startswith("Implement:"):
        sig = instruction[10:].strip()
        
        # Variation 1: "Write a function..."
        variations.append(f"Write a C function: {sig}")
        
        # Variation 2: "Create..."
        variations.append(f"Create: {sig}")
        
        # Variation 3: Just signature
        variations.append(sig)
    
    return variations


def augment_pair(pair: Dict[str, Any], max_variations: int = 2) -> List[Dict[str, Any]]:
    """Generate augmented versions of a training pair."""
    results = []
    
    instructions = augment_instruction(pair["instruction"])[:max_variations]
    
    for instr in instructions:
        new_pair = {
            "instruction": instr,
            "output": pair["output"],
            "metadata": {
                **pair.get("metadata", {}),
                "augmented": instr != pair["instruction"]
            }
        }
        results.append(new_pair)
    
    return results


# =============================================================================
# COMPILER (Optimized)
# =============================================================================

class FastCompiler:
    """Optimized C to WAT compiler."""
    
    CLANG_PATH = os.environ.get("CLANG_PATH", "clang")
    
    WRAPPER = '''
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
typedef unsigned long long uint64_t;
typedef unsigned long size_t;
typedef uint8_t BYTE;
typedef uint32_t WORD;
#ifndef NULL
#define NULL ((void*)0)
#endif
#define WASM_EXPORT __attribute__((visibility("default")))

{code}
'''
    
    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="zerolang_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
    
    def compile(self, code: str, func_name: str) -> Optional[str]:
        """Compile C code to WAT."""
        self._counter += 1
        base = f"{func_name}_{self._counter}"
        
        c_file = self.work_dir / f"{base}.c"
        wasm_file = self.work_dir / f"{base}.wasm"
        
        # Add WASM_EXPORT if not present
        if "WASM_EXPORT" not in code:
            # Find function definition and add export
            code = re.sub(
                r'^(\s*)((?:static\s+|inline\s+)*\w+[\s*]+' + re.escape(func_name) + r'\s*\()',
                r'\1WASM_EXPORT \2',
                code,
                flags=re.MULTILINE
            )
        
        wrapped = self.WRAPPER.format(code=code)
        c_file.write_text(wrapped)
        
        # Compile
        result = subprocess.run(
            [
                self.CLANG_PATH,
                "--target=wasm32",
                "-O2",
                "-nostdlib",
                "-fuse-ld=lld",
                "-Wl,--no-entry",
                "-Wl,--export-all",
                "-o", str(wasm_file),
                str(c_file)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        # Convert to WAT
        wat_result = subprocess.run(
            ["wasm-tools", "print", str(wasm_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if wat_result.returncode != 0:
            return None
        
        # Cleanup
        c_file.unlink(missing_ok=True)
        wasm_file.unlink(missing_ok=True)
        
        return wat_result.stdout
    
    def cleanup(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


# =============================================================================
# FUNCTION EXTRACTOR (Optimized)
# =============================================================================

class FunctionExtractor:
    """Extract C functions from source files."""
    
    FUNC_PATTERN = re.compile(
        r'^\s*'
        r'((?:static\s+|inline\s+|extern\s+)*'
        r'(?:const\s+)?'
        r'\w+(?:\s*\*+\s*|\s+)'
        r'(\w+)\s*'
        r'\([^)]*\)\s*'
        r')\{',
        re.MULTILINE
    )
    
    DOC_BLOCK = re.compile(r'/\*\*\s*(.*?)\*/', re.DOTALL)
    DOC_LINE = re.compile(r'((?:///[^\n]*\n)+)', re.MULTILINE)
    
    SKIP_NAMES = {'if', 'while', 'for', 'switch', 'sizeof', 'return', 'main', 'test'}
    SKIP_DIRS = {'test', 'tests', 'build', 'cmake', '.git', 'example', 'examples', 'bench'}
    
    def extract_from_repo(self, repo_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Extract all functions from a repository."""
        for ext in ['*.c', '*.h']:
            for file_path in repo_path.rglob(ext):
                if any(skip in file_path.parts for skip in self.SKIP_DIRS):
                    continue
                
                yield from self.extract_from_file(file_path, repo_path)
    
    def extract_from_file(self, file_path: Path, repo_path: Path) -> Generator[Dict[str, Any], None, None]:
        """Extract functions from a single file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return
        
        # Find doc comments
        doc_positions = []
        
        for match in self.DOC_BLOCK.finditer(content):
            doc = match.group(1)
            doc = re.sub(r'\n\s*\*\s*', ' ', doc)
            doc = re.sub(r'@\w+\s*', '', doc)
            doc = doc.strip()
            if doc:
                doc_positions.append((match.end(), doc))
        
        for match in self.DOC_LINE.finditer(content):
            lines = match.group(1).strip().split('\n')
            doc = ' '.join(line.strip()[3:].strip() for line in lines)
            if doc:
                doc_positions.append((match.end(), doc))
        
        # Find functions
        for match in self.FUNC_PATTERN.finditer(content):
            sig = match.group(1).strip()
            name = match.group(2)
            start = match.start()
            
            if name in self.SKIP_NAMES:
                continue
            
            # Find doc comment
            doc = None
            for doc_end, doc_text in doc_positions:
                gap = content[doc_end:start].strip()
                if len(gap) < 50 and '{' not in gap:
                    doc = doc_text
                    break
            
            # Fallback to signature
            if not doc:
                clean_sig = re.sub(r'\b(static|inline|extern)\s+', '', sig)
                clean_sig = ' '.join(clean_sig.split())
                doc = f"Implement: {clean_sig}"
            
            # Extract body
            body = self._extract_body(content[start:])
            if not body:
                continue
            
            # Quality check
            if body.count('\n') > 60:
                continue
            if self._has_complex_deps(body):
                continue
            
            yield {
                "name": name,
                "signature": sig,
                "body": body,
                "instruction": doc,
                "file": str(file_path.relative_to(repo_path)),
            }
    
    def _extract_body(self, content: str) -> Optional[str]:
        """Extract function body from opening brace."""
        brace_start = content.find('{')
        if brace_start == -1:
            return None
        
        depth = 0
        for i, char in enumerate(content[brace_start:], start=brace_start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return content[:i + 1]
        
        return None
    
    def _has_complex_deps(self, body: str) -> bool:
        """Check for complex dependencies."""
        patterns = [
            r'#include\s*[<"]',
            r'\basm\b',
            r'\b__attribute__',
            r'\bvolatile\b',
            r'\bgoto\b',
            r'\bextern\s+',
        ]
        return any(re.search(p, body) for p in patterns)


# =============================================================================
# MAIN COLLECTOR
# =============================================================================

@dataclass
class CollectionStats:
    repos_processed: int = 0
    repos_failed: int = 0
    functions_found: int = 0
    functions_compiled: int = 0
    pairs_after_filter: int = 0
    pairs_after_augment: int = 0
    duplicates_skipped: int = 0
    start_time: float = field(default_factory=time.time)
    
    def elapsed(self) -> str:
        secs = int(time.time() - self.start_time)
        return f"{secs // 60}m {secs % 60}s"


class LargeScaleCollector:
    """Large-scale data collection orchestrator."""
    
    def __init__(
        self,
        output_path: Path,
        target_pairs: int = 10000,
        quality_config: Optional[QualityConfig] = None,
        augment: bool = True,
        max_workers: int = 4,
    ):
        self.output_path = output_path
        self.target_pairs = target_pairs
        self.quality_config = quality_config or QualityConfig()
        self.augment = augment
        self.max_workers = max_workers
        
        self.stats = CollectionStats()
        self.seen_hashes: Set[str] = set()
        self.extractor = FunctionExtractor()
        
        # Load existing data for deduplication
        self._load_existing()
    
    def _load_existing(self):
        """Load existing pairs for deduplication."""
        if self.output_path.exists():
            print(f"[INFO] Loading existing data from {self.output_path}...")
            with open(self.output_path) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        h = self._hash(record['instruction'], record['output'])
                        self.seen_hashes.add(h)
            print(f"[INFO] Loaded {len(self.seen_hashes)} existing pairs")
    
    def _hash(self, instruction: str, output: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(f"{instruction}||{output}".encode()).hexdigest()
    
    def process_repo(self, repo_url: str) -> List[Dict[str, Any]]:
        """Process a single repository."""
        pairs = []
        repo_name = repo_url.rstrip('/').split('/')[-1]
        
        # Clone
        temp_dir = Path(tempfile.mkdtemp(prefix=f"zerolang_{repo_name}_"))
        try:
            result = subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(temp_dir / repo_name)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                self.stats.repos_failed += 1
                return []
            
            repo_path = temp_dir / repo_name
            compiler = FastCompiler()
            
            try:
                for func_info in self.extractor.extract_from_repo(repo_path):
                    self.stats.functions_found += 1
                    
                    wat = compiler.compile(func_info['body'], func_info['name'])
                    if not wat:
                        continue
                    
                    self.stats.functions_compiled += 1
                    
                    pair = {
                        "instruction": func_info['instruction'],
                        "output": wat,
                        "metadata": {
                            "function_name": func_info['name'],
                            "source_file": func_info['file'],
                            "repo_url": repo_url,
                            "signature": func_info['signature'],
                        }
                    }
                    
                    # Quality filter
                    if not passes_quality_filter(pair, self.quality_config):
                        continue
                    
                    self.stats.pairs_after_filter += 1
                    
                    # Dedup
                    h = self._hash(pair['instruction'], pair['output'])
                    if h in self.seen_hashes:
                        self.stats.duplicates_skipped += 1
                        continue
                    self.seen_hashes.add(h)
                    
                    pairs.append(pair)
                    
                    # Augmentation
                    if self.augment:
                        for aug_pair in augment_pair(pair)[1:]:  # Skip original
                            aug_h = self._hash(aug_pair['instruction'], aug_pair['output'])
                            if aug_h not in self.seen_hashes:
                                self.seen_hashes.add(aug_h)
                                pairs.append(aug_pair)
                                self.stats.pairs_after_augment += 1
            
            finally:
                compiler.cleanup()
        
        except Exception as e:
            print(f"[ERROR] {repo_name}: {e}")
            self.stats.repos_failed += 1
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        self.stats.repos_processed += 1
        return pairs
    
    def collect(self) -> int:
        """Run the collection process."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten repo list
        all_repos = []
        for tier, repos in REPOS.items():
            all_repos.extend(repos)
        
        print(f"\n{'='*60}")
        print(f"ZeroLang Large-Scale Collection")
        print(f"{'='*60}")
        print(f"Target: {self.target_pairs} pairs")
        print(f"Repos: {len(all_repos)}")
        print(f"Workers: {self.max_workers}")
        print(f"Output: {self.output_path}")
        print(f"{'='*60}\n")
        
        total_pairs = len(self.seen_hashes)
        
        with open(self.output_path, 'a') as f:
            for i, repo_url in enumerate(all_repos, 1):
                if total_pairs >= self.target_pairs:
                    print(f"\n[INFO] Target reached: {total_pairs} pairs")
                    break
                
                repo_name = repo_url.split('/')[-1]
                print(f"\n[{i}/{len(all_repos)}] Processing {repo_name}...")
                
                pairs = self.process_repo(repo_url)
                
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                    total_pairs += 1
                
                f.flush()
                
                print(f"  → +{len(pairs)} pairs (total: {total_pairs})")
                print(f"  → Stats: found={self.stats.functions_found}, compiled={self.stats.functions_compiled}")
        
        self._print_summary(total_pairs)
        return total_pairs
    
    def _print_summary(self, total_pairs: int):
        """Print collection summary."""
        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"{'='*60}")
        print(f"Time: {self.stats.elapsed()}")
        print(f"Repos processed: {self.stats.repos_processed}")
        print(f"Repos failed: {self.stats.repos_failed}")
        print(f"Functions found: {self.stats.functions_found}")
        print(f"Functions compiled: {self.stats.functions_compiled}")
        print(f"Pairs after filter: {self.stats.pairs_after_filter}")
        print(f"Pairs from augmentation: {self.stats.pairs_after_augment}")
        print(f"Duplicates skipped: {self.stats.duplicates_skipped}")
        print(f"{'='*60}")
        print(f"Total pairs: {total_pairs}")
        print(f"Output: {self.output_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Large-scale data collection for ZeroLang")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/large_dataset.jsonl"),
        help="Output file path"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=10000,
        help="Target number of pairs"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file"
    )
    
    args = parser.parse_args()
    
    # Clear output if not resuming
    if not args.resume and args.output.exists():
        args.output.unlink()
    
    collector = LargeScaleCollector(
        output_path=args.output,
        target_pairs=args.target,
        augment=not args.no_augment,
        max_workers=args.workers,
    )
    
    collector.collect()


if __name__ == "__main__":
    main()
