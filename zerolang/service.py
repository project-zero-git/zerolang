#!/usr/bin/env python3
"""
ZeroLang Service - End-to-end code generation and execution

Architecture:
1. Planner: Breaks down complex tasks into simple functions
2. Generator: ZeroLang model generates WAT for each function  
3. Assembler: Combines WAT modules
4. Runtime: Executes WASM

Usage:
    # Start service (connects to Gradio API)
    python zerolang/service.py --api-url https://xxx.gradio.live
    
    # Or with local model
    python zerolang/service.py --model-path models/zerolang-qwen-coder-14b-large
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

# Optional: OpenAI for planning
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Optional: Gradio client
try:
    from gradio_client import Client
    HAS_GRADIO_CLIENT = True
except ImportError:
    HAS_GRADIO_CLIENT = False


@dataclass
class Function:
    """A function to be generated."""
    name: str
    instruction: str
    wat: Optional[str] = None


@dataclass  
class Program:
    """A complete program with multiple functions."""
    name: str
    description: str
    functions: List[Function]
    wat_module: Optional[str] = None


class Planner:
    """
    Breaks down complex tasks into simple functions.
    Uses GPT-4/Claude for complex tasks, or rule-based for simple ones.
    """
    
    # Simple patterns that don't need LLM planning
    SIMPLE_PATTERNS = [
        (r"^implement:?\s*(.+)$", lambda m: [m.group(1)]),
        (r"^(int|void|float|double|char)\s+\w+\s*\(", lambda m: [m.group(0)]),
    ]
    
    def __init__(self, openai_key: Optional[str] = None):
        self.openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        
    def plan(self, task: str) -> List[str]:
        """Break down a task into function instructions."""
        task_lower = task.lower().strip()
        
        # Check simple patterns first
        for pattern, extractor in self.SIMPLE_PATTERNS:
            match = re.match(pattern, task_lower, re.IGNORECASE)
            if match:
                return extractor(match)
        
        # Complex task - use LLM if available
        if self.openai_key and HAS_OPENAI:
            return self._plan_with_llm(task)
        
        # Fallback: treat as single instruction
        return [f"Implement: {task}"]
    
    def _plan_with_llm(self, task: str) -> List[str]:
        """Use GPT-4 to break down complex tasks."""
        client = openai.OpenAI(api_key=self.openai_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a programming task planner. Break down the user's request into simple C function signatures.

Output format: JSON array of function signatures.
Each function should be standalone and compilable to WebAssembly.
Focus on pure computation - no I/O, no system calls.

Example:
User: "Calculator with add, subtract, multiply"
Output: ["int add(int a, int b)", "int subtract(int a, int b)", "int multiply(int a, int b)"]

User: "Factorial and fibonacci"
Output: ["int factorial(int n)", "int fibonacci(int n)"]"""
                },
                {"role": "user", "content": task}
            ],
            temperature=0.2,
        )
        
        try:
            content = response.choices[0].message.content
            # Extract JSON array
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                functions = json.loads(match.group())
                return [f"Implement: {f}" for f in functions]
        except Exception as e:
            print(f"[WARN] LLM planning failed: {e}")
        
        return [f"Implement: {task}"]


class Generator:
    """
    Generates WAT code using ZeroLang model.
    Can use either local model or remote API.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        model_path: Optional[Path] = None,
    ):
        self.api_url = api_url
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        
        if api_url and HAS_GRADIO_CLIENT:
            self.client = Client(api_url)
        else:
            self.client = None
    
    def generate(self, instruction: str) -> str:
        """Generate WAT code from instruction."""
        if self.client:
            return self._generate_api(instruction)
        elif self.model_path:
            return self._generate_local(instruction)
        else:
            raise RuntimeError("No API URL or model path configured")
    
    def _generate_api(self, instruction: str) -> str:
        """Generate using remote Gradio API."""
        result = self.client.predict(instruction, api_name="/predict")
        return result
    
    def _generate_local(self, instruction: str) -> str:
        """Generate using local model."""
        if self._model is None:
            self._load_model()
        
        import torch
        
        messages = [
            {"role": "system", "content": "You are ZeroLang, an AI that generates WebAssembly (WAT) code."},
            {"role": "user", "content": instruction},
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
            )
        
        wat = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return wat
    
    def _load_model(self):
        """Load local model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        print(f"[INFO] Loading model from {self.model_path}...")
        
        # Load config to get base model
        import json
        config_path = self.model_path / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        base_model_name = config.get("base_model_name_or_path")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base_model, self.model_path)
        self._model.eval()
        
        print(f"[INFO] Model loaded!")


class Assembler:
    """Combines multiple WAT functions into a single module."""
    
    def assemble(self, functions: List[Function]) -> str:
        """Combine WAT functions into a single module."""
        # Extract function bodies from each WAT
        func_bodies = []
        exports = []
        
        for func in functions:
            if not func.wat:
                continue
            
            # Extract the main function from the module
            # This is a simplified extraction - real implementation would parse WAT properly
            wat = func.wat
            
            # Find func definitions
            func_match = re.search(
                r'\(func \$(\w+).*?\(result.*?\).*?(?=\(func|\(@|\)$)',
                wat, 
                re.DOTALL
            )
            
            if func_match:
                func_bodies.append(func_match.group(0))
                exports.append(f'(export "{func.name}" (func ${func_match.group(1)}))')
        
        # Build combined module
        module = "(module\n"
        module += "  (memory 1)\n"
        
        for body in func_bodies:
            module += f"  {body}\n"
        
        for export in exports:
            module += f"  {export}\n"
        
        module += ")\n"
        
        return module


class Runtime:
    """Executes WASM code."""
    
    def __init__(self):
        self.work_dir = Path(tempfile.mkdtemp(prefix="zerolang_runtime_"))
    
    def compile_wat(self, wat: str) -> Path:
        """Compile WAT to WASM."""
        wat_file = self.work_dir / "program.wat"
        wasm_file = self.work_dir / "program.wasm"
        
        wat_file.write_text(wat)
        
        # Use wat2wasm or wasm-tools
        result = subprocess.run(
            ["wasm-tools", "parse", str(wat_file), "-o", str(wasm_file)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"WAT compilation failed: {result.stderr}")
        
        return wasm_file
    
    def execute(self, wasm_file: Path, func_name: str, *args) -> Any:
        """Execute a WASM function."""
        # Use wasmtime CLI for now
        # In production, use wasmtime-py bindings
        
        args_str = " ".join(str(a) for a in args)
        
        result = subprocess.run(
            ["wasmtime", str(wasm_file), "--invoke", func_name, *[str(a) for a in args]],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"WASM execution failed: {result.stderr}")
        
        return result.stdout.strip()
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)


class ZeroLangService:
    """
    Main service that orchestrates the full pipeline.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        model_path: Optional[Path] = None,
        openai_key: Optional[str] = None,
    ):
        self.planner = Planner(openai_key)
        self.generator = Generator(api_url, model_path)
        self.assembler = Assembler()
        self.runtime = Runtime()
    
    def generate_program(self, task: str) -> Program:
        """Generate a complete program from a task description."""
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print('='*60)
        
        # Step 1: Plan
        print("\n[1/3] Planning...")
        instructions = self.planner.plan(task)
        print(f"  Functions to generate: {len(instructions)}")
        for i, instr in enumerate(instructions, 1):
            print(f"    {i}. {instr}")
        
        # Step 2: Generate WAT for each function
        print("\n[2/3] Generating WAT...")
        functions = []
        for instr in instructions:
            name = self._extract_func_name(instr)
            print(f"  Generating {name}...")
            
            wat = self.generator.generate(instr)
            func = Function(name=name, instruction=instr, wat=wat)
            functions.append(func)
            
            print(f"    ✓ Generated {len(wat)} chars")
        
        # Step 3: Assemble
        print("\n[3/3] Assembling...")
        program = Program(
            name=self._extract_program_name(task),
            description=task,
            functions=functions,
        )
        
        if len(functions) > 1:
            program.wat_module = self.assembler.assemble(functions)
        else:
            program.wat_module = functions[0].wat if functions else None
        
        print(f"  ✓ Assembled module ({len(program.wat_module or '')} chars)")
        
        return program
    
    def run(self, task: str, func_name: Optional[str] = None, *args) -> str:
        """Generate and execute a program."""
        program = self.generate_program(task)
        
        if not program.wat_module:
            raise RuntimeError("No WAT generated")
        
        # Compile
        print("\n[4/4] Compiling & Executing...")
        try:
            wasm_file = self.runtime.compile_wat(program.wat_module)
            
            # Execute
            if func_name and args:
                result = self.runtime.execute(wasm_file, func_name, *args)
                print(f"  ✓ Result: {result}")
                return result
            else:
                print(f"  ✓ Compiled successfully")
                return program.wat_module
                
        except Exception as e:
            print(f"  ✗ Execution failed: {e}")
            # Return WAT even if execution fails
            return program.wat_module
    
    def _extract_func_name(self, instruction: str) -> str:
        """Extract function name from instruction."""
        # Match: "Implement: int func_name(...)"
        match = re.search(r'(\w+)\s*\(', instruction)
        if match:
            return match.group(1)
        return "unknown"
    
    def _extract_program_name(self, task: str) -> str:
        """Extract program name from task."""
        # Simple: first few words
        words = task.split()[:3]
        return "_".join(w.lower() for w in words if w.isalnum())
    
    def cleanup(self):
        """Clean up resources."""
        self.runtime.cleanup()


def main():
    parser = argparse.ArgumentParser(description="ZeroLang Service")
    parser.add_argument(
        "--api-url",
        type=str,
        help="Gradio API URL (e.g., https://xxx.gradio.live)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to local model"
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key for planning (optional)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to execute"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )
    
    args = parser.parse_args()
    
    if not args.api_url and not args.model_path:
        parser.error("Either --api-url or --model-path is required")
    
    service = ZeroLangService(
        api_url=args.api_url,
        model_path=args.model_path,
        openai_key=args.openai_key,
    )
    
    try:
        if args.interactive:
            print("\n" + "="*60)
            print("ZeroLang Interactive Mode")
            print("Type 'quit' to exit")
            print("="*60 + "\n")
            
            while True:
                try:
                    task = input("Task: ").strip()
                    if task.lower() in ['quit', 'exit', 'q']:
                        break
                    if not task:
                        continue
                    
                    result = service.run(task)
                    print(f"\n{result}\n")
                    
                except KeyboardInterrupt:
                    break
            
            print("\nGoodbye!")
        
        elif args.task:
            result = service.run(args.task)
            print(f"\n{result}")
        
        else:
            # Demo
            demo_tasks = [
                "Implement: int add(int a, int b)",
                "Implement: int factorial(int n)",
            ]
            
            for task in demo_tasks:
                service.run(task)
                print()
    
    finally:
        service.cleanup()


if __name__ == "__main__":
    main()
