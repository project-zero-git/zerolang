#!/usr/bin/env python3
"""
ZeroLang Data Pipeline - Phase 1: The Great Transpilation

This script generates training data for the ZeroLang model by:
1. Cloning GitHub repositories
2. Finding C functions with documentation
3. Compiling them to WAT (WebAssembly Text Format) using LLVM clang
4. Saving [instruction, output] pairs to JSONL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, List, Set


@dataclass
class CFunction:
    """Represents an extracted C function with its metadata."""
    name: str
    signature: str
    body: str
    doc_comment: str
    file_path: str


@dataclass
class TrainingPair:
    """A single training example for the model."""
    instruction: str
    output: str
    metadata: dict


class GitHubCloner:
    """Handles cloning and cleanup of GitHub repositories."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.mkdtemp(prefix="zerolang_"))
    
    def clone(self, repo_url: str) -> Path:
        """Clone a GitHub repository and return its local path."""
        repo_name = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        
        clone_path = self.base_dir / repo_name
        
        if clone_path.exists():
            print(f"[INFO] Repository already exists at {clone_path}, removing...")
            shutil.rmtree(clone_path)
        
        print(f"[INFO] Cloning {repo_url} to {clone_path}...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo_url, str(clone_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repository: {result.stderr}")
        
        print(f"[INFO] Successfully cloned {repo_name}")
        return clone_path
    
    def cleanup(self) -> None:
        """Remove the temporary directory and all cloned repos."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
            print(f"[INFO] Cleaned up {self.base_dir}")


class CFunctionExtractor:
    """Extracts functions and their documentation from C source files."""
    
    # Match C function definitions
    FUNCTION_PATTERN = re.compile(
        r'^\s*'
        r'((?:static\s+|inline\s+|extern\s+)*'  # optional modifiers
        r'(?:const\s+)?'  # optional const
        r'\w+(?:\s*\*+\s*|\s+)'  # return type
        r'(\w+)\s*'  # function name
        r'\([^)]*\)\s*'  # parameters
        r')\{',  # opening brace
        re.MULTILINE
    )
    
    # Match doxygen-style comments
    DOXYGEN_BLOCK = re.compile(
        r'/\*\*\s*(.*?)\*/',
        re.DOTALL
    )
    
    DOXYGEN_LINE = re.compile(
        r'((?:///[^\n]*\n)+)',
        re.MULTILINE
    )
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
    
    def find_source_files(self) -> Generator[Path, None, None]:
        """Walk through the repository and yield all .c and .h files."""
        for ext in ["*.c", "*.h"]:
            for c_file in self.repo_path.rglob(ext):
                # Skip test files and build artifacts
                if any(skip in c_file.parts for skip in ["test", "tests", "build", "cmake", ".git"]):
                    continue
                yield c_file
    
    def extract_functions(self, file_path: Path, require_doc: bool = False) -> Generator[CFunction, None, None]:
        """Extract functions from a C file.
        
        Args:
            file_path: Path to the C source file
            require_doc: If True, only extract functions with doc comments.
                        If False, use function signature as instruction when no doc.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[WARN] Could not decode {file_path}, skipping...")
            return
        
        # Find all doxygen block comments and their positions
        doc_positions = []
        
        # Block comments /** ... */
        for match in self.DOXYGEN_BLOCK.finditer(content):
            doc_text = match.group(1)
            # Clean up the doc text
            doc_text = re.sub(r'\n\s*\*\s*', ' ', doc_text)
            doc_text = re.sub(r'@\w+\s*', '', doc_text)  # Remove @param, @return etc
            doc_text = doc_text.strip()
            if doc_text:
                doc_positions.append((match.end(), doc_text))
        
        # Line comments /// ...
        for match in self.DOXYGEN_LINE.finditer(content):
            lines = match.group(1).strip().split('\n')
            doc_text = ' '.join(line.strip()[3:].strip() for line in lines)
            if doc_text:
                doc_positions.append((match.end(), doc_text))
        
        # Find functions
        for match in self.FUNCTION_PATTERN.finditer(content):
            fn_signature = match.group(1).strip()
            fn_name = match.group(2)
            fn_start = match.start()
            
            # Skip common non-function patterns
            if fn_name in ['if', 'while', 'for', 'switch', 'sizeof', 'return']:
                continue
            
            # Find the closest preceding doc comment
            doc_comment = ""
            for doc_end, doc_text in doc_positions:
                # Check if doc comment is close to function (within 50 chars, allowing for whitespace)
                gap = content[doc_end:fn_start].strip()
                if len(gap) < 50 and not gap.count('{'):
                    doc_comment = doc_text
                    break
            
            # If no doc comment, use signature as instruction (unless require_doc is True)
            if not doc_comment:
                if require_doc:
                    continue
                # Use clean signature as instruction
                doc_comment = self._signature_to_instruction(fn_signature)
            
            # Extract function body
            fn_body = self._extract_function_body(content[match.start():])
            
            if fn_body and self._is_simple_function(fn_body):
                yield CFunction(
                    name=fn_name,
                    signature=fn_signature,
                    body=fn_body,
                    doc_comment=doc_comment,
                    file_path=str(file_path.relative_to(self.repo_path))
                )
    
    def _signature_to_instruction(self, signature: str) -> str:
        """Convert a C function signature to a natural language-like instruction."""
        # Clean up the signature
        sig = signature.strip()
        # Remove storage class specifiers
        sig = re.sub(r'\b(static|inline|extern)\s+', '', sig)
        # Normalize whitespace
        sig = ' '.join(sig.split())
        return f"Implement: {sig}"
    
    def _extract_function_body(self, content: str) -> Optional[str]:
        """Extract content from opening brace to matching closing brace."""
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
    
    def _is_simple_function(self, fn_body: str) -> bool:
        """Check if a C function is simple enough to compile standalone."""
        complex_patterns = [
            r'#include\s*[<"]',  # includes inside function
            r'\basm\b',  # inline assembly
            r'\b__attribute__',  # compiler attributes
            r'\bvolatile\b',  # volatile (usually hardware)
            r'\bgoto\b',  # goto statements
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, fn_body):
                return False
        
        if fn_body.count('\n') > 50:
            return False
        
        return True


class CWatCompiler:
    """Compiles C functions to WebAssembly Text Format using LLVM clang."""
    
    # Use LLVM clang from homebrew (has WASM support)
    CLANG_PATH = "/opt/homebrew/opt/llvm/bin/clang"
    
    WRAPPER_TEMPLATE = '''
// Minimal WASM-compatible C wrapper
#define WASM_EXPORT __attribute__((visibility("default")))

// Standard integer types for standalone compilation
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
typedef unsigned long long uint64_t;
typedef unsigned long size_t;

// Common type aliases
typedef uint8_t BYTE;
typedef uint32_t WORD;
typedef uint64_t DWORD;

// NULL definition
#ifndef NULL
#define NULL ((void*)0)
#endif

{function}
'''
    
    def __init__(self, work_dir: Optional[Path] = None, verbose: bool = False):
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="zerolang_compile_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Check if LLVM clang is available
        if not Path(self.CLANG_PATH).exists():
            # Try system clang as fallback
            self.CLANG_PATH = "clang"
    
    def compile_to_wat(self, fn: CFunction) -> Optional[str]:
        """Compile a C function to WAT format."""
        source_file = self.work_dir / f"{fn.name}.c"
        wasm_file = self.work_dir / f"{fn.name}.wasm"
        
        # Make function exportable
        modified_body = fn.body
        if not fn.body.strip().startswith("WASM_EXPORT"):
            modified_body = f"WASM_EXPORT {fn.body}"
        
        wrapped_source = self.WRAPPER_TEMPLATE.format(function=modified_body)
        source_file.write_text(wrapped_source)
        
        # Compile to WASM using LLVM clang with lld linker
        compile_result = subprocess.run(
            [
                self.CLANG_PATH,
                "--target=wasm32",
                "-O2",
                "-nostdlib",
                "-fuse-ld=lld",
                "-Wl,--no-entry",
                "-Wl,--export-all",
                "-o", str(wasm_file),
                str(source_file)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode != 0:
            if self.verbose:
                print(f"[DEBUG] Compilation failed for {fn.name}:")
                for line in compile_result.stderr.split('\n')[:3]:
                    if line.strip():
                        print(f"        {line}")
            return None
        
        return self._wasm_to_wat(wasm_file)
    
    def _wasm_to_wat(self, wasm_file: Path) -> Optional[str]:
        """Convert WASM binary to WAT text format."""
        wat_result = subprocess.run(
            ["wasm-tools", "print", str(wasm_file)],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if wat_result.returncode == 0:
            return wat_result.stdout
        
        return None
    
    def cleanup(self) -> None:
        """Remove temporary compilation files."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)


@dataclass
class DatasetStats:
    """Statistics for dataset generation."""
    repos_processed: int = 0
    files_scanned: int = 0
    functions_found: int = 0
    functions_compiled: int = 0
    pairs_generated: int = 0
    duplicates_skipped: int = 0
    
    def to_dict(self) -> dict:
        return {
            "repos_processed": self.repos_processed,
            "files_scanned": self.files_scanned,
            "functions_found": self.functions_found,
            "functions_compiled": self.functions_compiled,
            "pairs_generated": self.pairs_generated,
            "duplicates_skipped": self.duplicates_skipped
        }


class DatasetGenerator:
    """Orchestrates the full pipeline from repo to JSONL."""
    
    def __init__(self, output_path: Path, verbose: bool = False, require_doc: bool = False):
        self.output_path = output_path
        self.verbose = verbose
        self.require_doc = require_doc  # If False, use signature when no doc
        self.cloner = GitHubCloner()
        self.compiler = CWatCompiler(verbose=verbose)
        self.stats = DatasetStats()
        self.seen_hashes: Set[str] = set()
    
    def _get_content_hash(self, instruction: str, output: str) -> str:
        """Generate a hash for deduplication."""
        content = f"{instruction}||{output}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_repository(self, repo_url: str) -> Generator[TrainingPair, None, None]:
        """Process a single repository and yield training pairs."""
        try:
            repo_path = self.cloner.clone(repo_url)
            self.stats.repos_processed += 1
            
            extractor = CFunctionExtractor(repo_path)
            
            for source_file in extractor.find_source_files():
                self.stats.files_scanned += 1
                print(f"[INFO] Scanning {source_file}...")
                
                for fn in extractor.extract_functions(source_file, require_doc=self.require_doc):
                    self.stats.functions_found += 1
                    
                    print(f"[INFO] Compiling {fn.name}...")
                    wat_output = self.compiler.compile_to_wat(fn)
                    
                    if wat_output:
                        # Check for duplicates
                        content_hash = self._get_content_hash(fn.doc_comment, wat_output)
                        if content_hash in self.seen_hashes:
                            self.stats.duplicates_skipped += 1
                            continue
                        self.seen_hashes.add(content_hash)
                        
                        self.stats.functions_compiled += 1
                        self.stats.pairs_generated += 1
                        
                        yield TrainingPair(
                            instruction=fn.doc_comment,
                            output=wat_output,
                            metadata={
                                "function_name": fn.name,
                                "source_file": fn.file_path,
                                "repo_url": repo_url,
                                "signature": fn.signature
                            }
                        )
                    
        except Exception as e:
            print(f"[ERROR] Failed to process {repo_url}: {e}")
    
    def generate(self, repo_urls: List[str], append: bool = False) -> None:
        """Generate the full dataset from a list of repositories.
        
        Args:
            repo_urls: List of GitHub repository URLs
            append: If True, append to existing file instead of overwriting
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing hashes if appending
        if append and self.output_path.exists():
            print(f"[INFO] Loading existing data for deduplication...")
            with open(self.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        content_hash = self._get_content_hash(record['instruction'], record['output'])
                        self.seen_hashes.add(content_hash)
            print(f"[INFO] Loaded {len(self.seen_hashes)} existing pairs")
        
        mode = 'a' if append else 'w'
        with open(self.output_path, mode, encoding='utf-8') as f:
            for repo_url in repo_urls:
                print(f"\n{'='*60}")
                print(f"Processing: {repo_url}")
                print('='*60)
                
                for pair in self.process_repository(repo_url):
                    record = {
                        "instruction": pair.instruction,
                        "output": pair.output,
                        "metadata": pair.metadata
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    print(f"[SUCCESS] Generated pair for: {pair.metadata['function_name']}")
        
        self._print_stats()
        self._save_metadata()
    
    def _print_stats(self) -> None:
        """Print generation statistics."""
        print(f"\n{'='*60}")
        print("Generation Complete!")
        print('='*60)
        for key, value in self.stats.to_dict().items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print(f"\nOutput saved to: {self.output_path}")
    
    def _save_metadata(self) -> None:
        """Save dataset metadata to JSON file."""
        metadata_path = self.output_path.with_suffix('.meta.json')
        metadata = {
            "stats": self.stats.to_dict(),
            "language": "c",
            "output_file": str(self.output_path)
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
    
    def cleanup(self) -> None:
        """Clean up all temporary resources."""
        self.cloner.cleanup()
        self.compiler.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="ZeroLang Data Pipeline - Generate WASM training data from C repositories"
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        action="append",
        dest="repos",
        help="GitHub repository URL to process (can be specified multiple times)"
    )
    parser.add_argument(
        "--repo-list", "-l",
        type=Path,
        help="Path to a file containing repository URLs (one per line)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/training_data.jsonl"),
        help="Output path for the JSONL file (default: data/training_data.jsonl)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after processing"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show compilation errors for debugging"
    )
    parser.add_argument(
        "--require-doc",
        action="store_true",
        help="Only extract functions with doc comments (default: use signature when no doc)"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting"
    )
    
    args = parser.parse_args()
    
    # Collect repository URLs
    repo_urls: List[str] = []
    
    if args.repos:
        repo_urls.extend(args.repos)
    
    if args.repo_list and args.repo_list.exists():
        with open(args.repo_list, 'r') as f:
            repo_urls.extend(
                line.strip() for line in f 
                if line.strip() and not line.startswith('#')
            )
    
    if not repo_urls:
        parser.error("No repositories specified. Use --repo or --repo-list")
    
    # Run the pipeline
    generator = DatasetGenerator(args.output, verbose=args.verbose, require_doc=args.require_doc)
    
    try:
        generator.generate(repo_urls, append=args.append)
    finally:
        if not args.keep_temp:
            generator.cleanup()


if __name__ == "__main__":
    main()
