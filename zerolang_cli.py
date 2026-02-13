#!/usr/bin/env python3
"""
ZeroLang CLI - Natural Language to WebAssembly

Interactive console application for generating and executing WASM code.

Usage:
    python zerolang_cli.py --api <gradio-url>
    python zerolang_cli.py --api https://xxx.gradio.live
"""

from __future__ import annotations

import argparse
import sys
import re
from typing import Optional

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def colored(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.END}"
    return text


def print_banner():
    """Print the ZeroLang banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗███████╗██████╗  ██████╗ ██╗      █████╗ ███╗   ██╗ ║
║   ╚══███╔╝██╔════╝██╔══██╗██╔═══██╗██║     ██╔══██╗████╗  ██║ ║
║     ███╔╝ █████╗  ██████╔╝██║   ██║██║     ███████║██╔██╗ ██║ ║
║    ███╔╝  ██╔══╝  ██╔══██╗██║   ██║██║     ██╔══██║██║╚██╗██║ ║
║   ███████╗███████╗██║  ██║╚██████╔╝███████╗██║  ██║██║ ╚████║ ║
║   ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ║
║                                                               ║
║          Natural Language → WebAssembly Compiler              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(colored(banner, Colors.CYAN))


def print_help():
    """Print help information."""
    help_text = """
Commands:
  gen <instruction>    Generate WAT from instruction
  run <func(args)>     Run the last generated WAT
  genrun <instr> | <call>  Generate and run in one command
  wat                  Show the last generated WAT
  clear                Clear the screen
  help                 Show this help
  quit / exit          Exit the program

Examples:
  > gen Implement: int add(int a, int b)
  > run add(5, 3)
  
  > genrun Implement: int max(int a, int b) | max(10, 25)
  
  > gen Implement: int multiply(int x, int y)
  > run multiply(7, 6)
"""
    print(colored(help_text, Colors.DIM))


class ZeroLangCLI:
    """Interactive CLI for ZeroLang."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.last_wat: Optional[str] = None
        self.last_instruction: Optional[str] = None
        
        # Initialize components
        self._init_client()
        self._init_runtime()
    
    def _init_client(self):
        """Initialize Gradio client."""
        print(colored("[*] Connecting to API...", Colors.DIM))
        try:
            from gradio_client import Client
            self.client = Client(self.api_url, verbose=False)
            print(colored(f"[✓] Connected to {self.api_url}", Colors.GREEN))
        except Exception as e:
            print(colored(f"[✗] Failed to connect: {e}", Colors.RED))
            sys.exit(1)
    
    def _init_runtime(self):
        """Initialize WASM runtime."""
        print(colored("[*] Initializing WASM runtime...", Colors.DIM))
        try:
            from zrun.runtime import ZeroLangRuntime
            self.runtime = ZeroLangRuntime()
            print(colored("[✓] WASM runtime ready", Colors.GREEN))
        except Exception as e:
            print(colored(f"[!] WASM runtime not available: {e}", Colors.YELLOW))
            print(colored("    (Generate will work, but Run will fail)", Colors.DIM))
            self.runtime = None
    
    def generate(self, instruction: str) -> Optional[str]:
        """Generate WAT from instruction."""
        print(colored(f"\n[*] Generating WAT...", Colors.BLUE))
        
        try:
            wat = self.client.predict(instruction, api_name="/predict")
            self.last_wat = wat
            self.last_instruction = instruction
            
            # Count lines and show preview
            lines = wat.strip().split('\n')
            print(colored(f"[✓] Generated {len(lines)} lines of WAT", Colors.GREEN))
            
            # Show preview (first 10 lines)
            print(colored("\n--- WAT Preview ---", Colors.DIM))
            for line in lines[:10]:
                print(colored(line, Colors.CYAN))
            if len(lines) > 10:
                print(colored(f"... ({len(lines) - 10} more lines)", Colors.DIM))
            print(colored("-------------------", Colors.DIM))
            
            return wat
        
        except Exception as e:
            print(colored(f"[✗] Generation failed: {e}", Colors.RED))
            return None
    
    def run(self, call_str: str) -> Optional[int]:
        """Run a function from the last generated WAT."""
        if not self.last_wat:
            print(colored("[✗] No WAT generated yet. Use 'gen' first.", Colors.RED))
            return None
        
        if not self.runtime:
            print(colored("[✗] WASM runtime not available.", Colors.RED))
            return None
        
        # Parse function call
        match = re.match(r'(\w+)\s*\((.*)\)', call_str)
        if not match:
            print(colored(f"[✗] Invalid call format: {call_str}", Colors.RED))
            print(colored("    Expected: func_name(arg1, arg2, ...)", Colors.DIM))
            return None
        
        func_name = match.group(1)
        args_str = match.group(2).strip()
        args = [int(a.strip()) for a in args_str.split(',')] if args_str else []
        
        print(colored(f"\n[*] Executing {func_name}({', '.join(map(str, args))})...", Colors.BLUE))
        
        try:
            result = self.runtime.execute_wat(self.last_wat, func_name, args)
            print(colored(f"\n╔════════════════════════════════╗", Colors.GREEN))
            print(colored(f"║  Result: {str(result):>20}  ║", Colors.GREEN + Colors.BOLD))
            print(colored(f"╚════════════════════════════════╝", Colors.GREEN))
            return result
        
        except Exception as e:
            print(colored(f"[✗] Execution failed: {e}", Colors.RED))
            return None
    
    def genrun(self, combined: str):
        """Generate and run in one command."""
        if '|' not in combined:
            print(colored("[✗] Invalid format. Use: genrun <instruction> | <call>", Colors.RED))
            return
        
        instruction, call = combined.split('|', 1)
        instruction = instruction.strip()
        call = call.strip()
        
        wat = self.generate(instruction)
        if wat:
            self.run(call)
    
    def show_wat(self):
        """Show the last generated WAT."""
        if not self.last_wat:
            print(colored("[✗] No WAT generated yet.", Colors.RED))
            return
        
        print(colored(f"\n--- WAT for: {self.last_instruction} ---", Colors.DIM))
        print(colored(self.last_wat, Colors.CYAN))
        print(colored("--- End WAT ---", Colors.DIM))
    
    def run_interactive(self):
        """Run the interactive CLI loop."""
        print_banner()
        print_help()
        
        print(colored("\nReady! Type 'help' for commands.\n", Colors.GREEN))
        
        while True:
            try:
                # Prompt
                prompt = colored("zerolang> ", Colors.BOLD + Colors.BLUE)
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if cmd in ['quit', 'exit', 'q']:
                    print(colored("\nGoodbye! 👋", Colors.CYAN))
                    break
                
                elif cmd == 'help':
                    print_help()
                
                elif cmd == 'clear':
                    print('\033[2J\033[H')  # Clear screen
                    print_banner()
                
                elif cmd == 'gen':
                    if not arg:
                        print(colored("[✗] Usage: gen <instruction>", Colors.RED))
                    else:
                        self.generate(arg)
                
                elif cmd == 'run':
                    if not arg:
                        print(colored("[✗] Usage: run <func(args)>", Colors.RED))
                    else:
                        self.run(arg)
                
                elif cmd == 'genrun':
                    if not arg:
                        print(colored("[✗] Usage: genrun <instruction> | <call>", Colors.RED))
                    else:
                        self.genrun(arg)
                
                elif cmd == 'wat':
                    self.show_wat()
                
                else:
                    # Try to interpret as instruction directly
                    if user_input.startswith("Implement:") or user_input.startswith("implement:"):
                        self.generate(user_input)
                    else:
                        print(colored(f"[?] Unknown command: {cmd}", Colors.YELLOW))
                        print(colored("    Type 'help' for available commands.", Colors.DIM))
            
            except KeyboardInterrupt:
                print(colored("\n\nUse 'quit' to exit.", Colors.YELLOW))
            
            except EOFError:
                print(colored("\nGoodbye! 👋", Colors.CYAN))
                break


def main():
    parser = argparse.ArgumentParser(
        description="ZeroLang CLI - Natural Language to WebAssembly"
    )
    parser.add_argument(
        "--api", "-a",
        type=str,
        required=True,
        help="Gradio API URL (e.g., https://xxx.gradio.live)"
    )
    
    args = parser.parse_args()
    
    cli = ZeroLangCLI(args.api)
    cli.run_interactive()


if __name__ == "__main__":
    main()
