#!/usr/bin/env python3
"""
ZeroLang Data Pipeline - Post-processing

Handles:
1. Merging multiple JSONL files
2. Deduplication across files
3. Train/validation split
4. Statistics generation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import List, Set, Dict, Any
from collections import defaultdict


def compute_hash(record: Dict[str, Any]) -> str:
    """Compute a hash for deduplication."""
    content = f"{record['instruction']}||{record['output']}"
    return hashlib.md5(content.encode()).hexdigest()


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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def merge_and_deduplicate(
    input_files: List[Path],
    output_path: Path
) -> Dict[str, int]:
    """Merge multiple JSONL files and remove duplicates."""
    seen_hashes: Set[str] = set()
    all_records: List[Dict[str, Any]] = []
    stats = defaultdict(int)
    
    for file_path in input_files:
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue
        
        print(f"[INFO] Loading {file_path}...")
        records = load_jsonl(file_path)
        stats[f"loaded_from_{file_path.name}"] = len(records)
        
        for record in records:
            record_hash = compute_hash(record)
            if record_hash not in seen_hashes:
                seen_hashes.add(record_hash)
                all_records.append(record)
            else:
                stats["duplicates_removed"] += 1
    
    stats["total_unique"] = len(all_records)
    
    print(f"[INFO] Saving {len(all_records)} unique records to {output_path}...")
    save_jsonl(all_records, output_path)
    
    return dict(stats)


def create_train_val_split(
    input_path: Path,
    train_path: Path,
    val_path: Path,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, int]:
    """Split data into training and validation sets."""
    print(f"[INFO] Loading {input_path}...")
    records = load_jsonl(input_path)
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(records)
    
    # Calculate split point
    val_size = int(len(records) * val_ratio)
    train_size = len(records) - val_size
    
    train_records = records[:train_size]
    val_records = records[train_size:]
    
    print(f"[INFO] Saving {len(train_records)} training records to {train_path}...")
    save_jsonl(train_records, train_path)
    
    print(f"[INFO] Saving {len(val_records)} validation records to {val_path}...")
    save_jsonl(val_records, val_path)
    
    return {
        "total": len(records),
        "train": len(train_records),
        "validation": len(val_records),
        "val_ratio": val_ratio
    }


def compute_statistics(input_path: Path) -> Dict[str, Any]:
    """Compute detailed statistics about the dataset."""
    records = load_jsonl(input_path)
    
    stats = {
        "total_records": len(records),
        "by_language": defaultdict(int),
        "by_repo": defaultdict(int),
        "instruction_lengths": [],
        "output_lengths": [],
    }
    
    for record in records:
        # Count by language
        lang = record.get("metadata", {}).get("language", "unknown")
        stats["by_language"][lang] += 1
        
        # Count by repo
        repo = record.get("metadata", {}).get("repo_url", "unknown")
        repo_name = repo.split("/")[-1] if "/" in repo else repo
        stats["by_repo"][repo_name] += 1
        
        # Track lengths
        stats["instruction_lengths"].append(len(record.get("instruction", "")))
        stats["output_lengths"].append(len(record.get("output", "")))
    
    # Compute averages
    if stats["instruction_lengths"]:
        stats["avg_instruction_length"] = sum(stats["instruction_lengths"]) / len(stats["instruction_lengths"])
        stats["avg_output_length"] = sum(stats["output_lengths"]) / len(stats["output_lengths"])
    
    # Convert defaultdicts to regular dicts
    stats["by_language"] = dict(stats["by_language"])
    stats["by_repo"] = dict(stats["by_repo"])
    
    # Remove raw lists (too large)
    del stats["instruction_lengths"]
    del stats["output_lengths"]
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="ZeroLang Data Pipeline - Post-processing"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge and deduplicate JSONL files")
    merge_parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="Input JSONL files to merge"
    )
    merge_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output path for merged file"
    )
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Create train/validation split")
    split_parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSONL file"
    )
    split_parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Output path for training data"
    )
    split_parser.add_argument(
        "--val",
        type=Path,
        required=True,
        help="Output path for validation data"
    )
    split_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    split_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Compute dataset statistics")
    stats_parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSONL file"
    )
    stats_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for stats JSON (optional)"
    )
    
    args = parser.parse_args()
    
    if args.command == "merge":
        stats = merge_and_deduplicate(args.input_files, args.output)
        print("\n=== Merge Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == "split":
        stats = create_train_val_split(
            args.input_file,
            args.train,
            args.val,
            args.val_ratio,
            args.seed
        )
        print("\n=== Split Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == "stats":
        stats = compute_statistics(args.input_file)
        print("\n=== Dataset Statistics ===")
        print(json.dumps(stats, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nSaved to: {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
