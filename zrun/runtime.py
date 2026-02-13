#!/usr/bin/env python3
"""
ZeroLang Runtime (zrun)

Compiles WAT to WASM and executes it.

Usage:
    # Execute WAT file
    python zrun/runtime.py --wat output.wat --call "add(5, 3)"
    
    # Execute WAT string
    python zrun/runtime.py --wat-string "(module ...)" --call "add(5, 3)"
    
    # With ZeroLang API
    python zrun/runtime.py --api "https://xxx.gradio.live" --instruction "Implement: int add(int a, int b)" --call "add(5, 3)"
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Any


class WATCompiler:
    """Compiles WAT to WASM."""
    
    def __init__(self):
        self._check_tools()
    
    def _check_tools(self):
        """Check if required tools are installed."""
        # Check for wat2wasm (from wabt) or wasm-tools
        self.compiler = None
        
        # Try wasm-tools first
        result = subprocess.run(["wasm-tools", "--version"], capture_output=True)
        if result.returncode == 0:
            self.compiler = "wasm-tools"
            return
        
        # Try wat2wasm
        result = subprocess.run(["wat2wasm", "--version"], capture_output=True)
        if result.returncode == 0:
            self.compiler = "wat2wasm"
            return
        
        raise RuntimeError(
            "No WAT compiler found. Install one of:\n"
            "  - wasm-tools: cargo install wasm-tools\n"
            "  - wabt: brew install wabt"
        )
    
    def compile(self, wat: str) -> bytes:
        """Compile WAT string to WASM bytes."""
        with tempfile.NamedTemporaryFile(suffix=".wat", delete=False) as f:
            f.write(wat.encode())
            wat_path = Path(f.name)
        
        wasm_path = wat_path.with_suffix(".wasm")
        
        try:
            if self.compiler == "wasm-tools":
                result = subprocess.run(
                    ["wasm-tools", "parse", str(wat_path), "-o", str(wasm_path)],
                    capture_output=True,
                    text=True
                )
            else:  # wat2wasm
                result = subprocess.run(
                    ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
                    capture_output=True,
                    text=True
                )
            
            if result.returncode != 0:
                raise RuntimeError(f"WAT compilation failed:\n{result.stderr}")
            
            return wasm_path.read_bytes()
        
        finally:
            wat_path.unlink(missing_ok=True)
            wasm_path.unlink(missing_ok=True)


class WASMRunner:
    """Executes WASM modules."""
    
    def __init__(self):
        self._check_runtime()
    
    def _check_runtime(self):
        """Check if a WASM runtime is available."""
        # Try Python wasmtime module first (most reliable)
        try:
            import wasmtime
            self.runtime = "wasmtime-py"
            return
        except ImportError:
            pass
        
        # Try wasmtime CLI
        try:
            result = subprocess.run(["wasmtime", "--version"], capture_output=True)
            if result.returncode == 0:
                self.runtime = "wasmtime"
                return
        except FileNotFoundError:
            pass
        
        # Try wasmer
        try:
            result = subprocess.run(["wasmer", "--version"], capture_output=True)
            if result.returncode == 0:
                self.runtime = "wasmer"
                return
        except FileNotFoundError:
            pass
        
        raise RuntimeError(
            "No WASM runtime found. Install one of:\n"
            "  - wasmtime-py: pip install wasmtime\n"
            "  - wasmtime: curl https://wasmtime.dev/install.sh -sSf | bash\n"
            "  - wasmer: curl https://get.wasmer.io -sSf | sh"
        )
    
    def run(self, wasm: bytes, func_name: str, args: List[int]) -> Any:
        """Execute a function in the WASM module."""
        if self.runtime == "wasmtime-py":
            return self._run_wasmtime_py(wasm, func_name, args)
        else:
            return self._run_cli(wasm, func_name, args)
    
    def _run_wasmtime_py(self, wasm: bytes, func_name: str, args: List[int]) -> Any:
        """Run using Python wasmtime module."""
        import wasmtime
        
        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, wasm)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, module, [])
        
        # Get the function
        func = instance.exports(store).get(func_name)
        if func is None:
            # Try with underscore prefix
            func = instance.exports(store).get(f"_{func_name}")
        
        if func is None:
            available = [name for name in dir(instance.exports(store)) if not name.startswith('_')]
            raise RuntimeError(f"Function '{func_name}' not found. Available: {available}")
        
        return func(store, *args)
    
    def _run_cli(self, wasm: bytes, func_name: str, args: List[int]) -> Any:
        """Run using CLI runtime."""
        with tempfile.NamedTemporaryFile(suffix=".wasm", delete=False) as f:
            f.write(wasm)
            wasm_path = Path(f.name)
        
        try:
            args_str = " ".join(str(a) for a in args)
            
            if self.runtime == "wasmtime":
                cmd = ["wasmtime", "run", "--invoke", func_name, str(wasm_path), "--"] + [str(a) for a in args]
            else:  # wasmer
                cmd = ["wasmer", "run", str(wasm_path), "--invoke", func_name] + [str(a) for a in args]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"WASM execution failed:\n{result.stderr}")
            
            # Parse output
            output = result.stdout.strip()
            if output:
                try:
                    return int(output)
                except ValueError:
                    return output
            return None
        
        finally:
            wasm_path.unlink(missing_ok=True)


class ZeroLangRuntime:
    """Complete ZeroLang runtime: WAT → WASM → Execute."""
    
    def __init__(self):
        self.compiler = WATCompiler()
        self.runner = WASMRunner()
    
    def execute_wat(self, wat: str, func_name: str, args: List[int]) -> Any:
        """Execute a function from WAT code."""
        print(f"[zrun] Compiling WAT...")
        wasm = self.compiler.compile(wat)
        print(f"[zrun] Compiled to {len(wasm)} bytes WASM")
        
        print(f"[zrun] Executing {func_name}({', '.join(map(str, args))})...")
        result = self.runner.run(wasm, func_name, args)
        
        return result
    
    def execute_from_api(self, api_url: str, instruction: str, func_name: str, args: List[int]) -> Any:
        """Generate WAT from API and execute."""
        print(f"[zrun] Generating WAT from API...")
        
        # Use gradio_client if available
        try:
            from gradio_client import Client
            client = Client(api_url)
            wat = client.predict(instruction, api_name="/predict")
        except ImportError:
            # Fallback to requests
            import requests
            response = requests.post(
                f"{api_url}/api/predict",
                json={"data": [instruction]}
            )
            wat = response.json()["data"][0]
        
        print(f"[zrun] Generated {len(wat)} chars WAT")
        return self.execute_wat(wat, func_name, args)


def parse_call(call_str: str) -> tuple[str, List[int]]:
    """Parse a function call string like 'add(5, 3)' into (name, args)."""
    match = re.match(r'(\w+)\s*\((.*)\)', call_str)
    if not match:
        raise ValueError(f"Invalid call format: {call_str}. Expected: func_name(arg1, arg2, ...)")
    
    func_name = match.group(1)
    args_str = match.group(2).strip()
    
    if args_str:
        args = [int(a.strip()) for a in args_str.split(',')]
    else:
        args = []
    
    return func_name, args


def main():
    parser = argparse.ArgumentParser(description="ZeroLang Runtime - Execute WAT code")
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--wat", type=Path, help="Path to WAT file")
    input_group.add_argument("--wat-string", type=str, help="WAT code as string")
    input_group.add_argument("--api", type=str, help="ZeroLang API URL")
    
    # For API mode
    parser.add_argument("--instruction", type=str, help="Instruction for API (required with --api)")
    
    # Execution
    parser.add_argument("--call", type=str, required=True, help="Function call, e.g., 'add(5, 3)'")
    
    args = parser.parse_args()
    
    # Parse function call
    func_name, func_args = parse_call(args.call)
    
    # Initialize runtime
    runtime = ZeroLangRuntime()
    
    # Get WAT and execute
    if args.wat:
        wat = args.wat.read_text()
        result = runtime.execute_wat(wat, func_name, func_args)
    
    elif args.wat_string:
        result = runtime.execute_wat(args.wat_string, func_name, func_args)
    
    elif args.api:
        if not args.instruction:
            parser.error("--instruction is required with --api")
        result = runtime.execute_from_api(args.api, args.instruction, func_name, func_args)
    
    print(f"\n{'='*40}")
    print(f"Result: {result}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
