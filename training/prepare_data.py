#!/usr/bin/env python3
"""
ZeroLang Training Data Preparation

Converts JSONL pairs to instruction-tuning format for LLM fine-tuning.
Supports multiple formats: Alpaca, ChatML, ShareGPT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], file_path: Path) -> None:
    """Save records to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def convert_to_alpaca(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to Alpaca format.
    
    Format:
    {
        "instruction": "...",
        "input": "",
        "output": "..."
    }
    """
    converted = []
    for record in records:
        converted.append({
            "instruction": f"Generate WebAssembly (WAT) code for the following task:\n{record['instruction']}",
            "input": "",
            "output": record['output']
        })
    return converted


def convert_to_chatml(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to ChatML format (used by many models).
    
    Format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    system_prompt = (
        "You are ZeroLang, an AI that generates optimized WebAssembly (WAT) code. "
        "Given a natural language description of a function, output valid WAT code that implements it. "
        "Output only the WAT code, no explanations."
    )
    
    converted = []
    for record in records:
        converted.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": record['instruction']},
                {"role": "assistant", "content": record['output']}
            ]
        })
    return converted


def convert_to_completion(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert to simple completion format.
    
    Format:
    {
        "text": "<instruction>...</instruction><output>...</output>"
    }
    """
    converted = []
    for record in records:
        text = f"<instruction>{record['instruction']}</instruction>\n<output>\n{record['output']}\n</output>"
        converted.append({"text": text})
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert ZeroLang training data to various LLM formats"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["alpaca", "chatml", "completion"],
        default="chatml",
        help="Output format (default: chatml)"
    )
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading {args.input_file}...")
    records = load_jsonl(args.input_file)
    print(f"[INFO] Loaded {len(records)} records")
    
    print(f"[INFO] Converting to {args.format} format...")
    if args.format == "alpaca":
        converted = convert_to_alpaca(records)
    elif args.format == "chatml":
        converted = convert_to_chatml(records)
    elif args.format == "completion":
        converted = convert_to_completion(records)
    
    print(f"[INFO] Saving to {args.output}...")
    save_jsonl(converted, args.output)
    
    print(f"[SUCCESS] Converted {len(converted)} records to {args.format} format")
    
    # Show sample
    print("\n=== Sample Record ===")
    print(json.dumps(converted[0], indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
