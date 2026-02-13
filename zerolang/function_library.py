"""
ZeroLang Function Library

Pre-compiled WAT function cache for instant execution.
No LLM inference needed for known functions.

Architecture:
1. functions.json - Function definitions (signature, description, WAT)
2. FunctionLibrary - Semantic search + exact match
3. ZeroLangEngine - Cache hit → instant, Cache miss → LLM fallback
"""

import json
import os
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Function:
    """A pre-compiled function."""
    name: str
    signature: str
    description: str
    wat: str
    keywords: List[str]
    category: str


class FunctionLibrary:
    """
    Pre-compiled function library with semantic matching.
    
    Usage:
        lib = FunctionLibrary()
        lib.load("functions.json")
        
        # Exact match
        wat = lib.get("add")
        
        # Semantic search
        wat, score = lib.search("sum two numbers")
    """
    
    def __init__(self):
        self.functions: Dict[str, Function] = {}
        self.keywords_index: Dict[str, List[str]] = {}  # keyword → [func_names]
    
    def add(self, func: Function):
        """Add a function to the library."""
        self.functions[func.name] = func
        
        # Index keywords
        for kw in func.keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.keywords_index:
                self.keywords_index[kw_lower] = []
            self.keywords_index[kw_lower].append(func.name)
    
    def get(self, name: str) -> Optional[str]:
        """Get WAT by exact function name."""
        func = self.functions.get(name)
        return func.wat if func else None
    
    def search(self, query: str, threshold: float = 0.3) -> Optional[Tuple[Function, float]]:
        """
        Search for a function by natural language query.
        Returns (Function, score) or None if no match above threshold.
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        best_match = None
        best_score = 0.0
        
        for func in self.functions.values():
            score = self._match_score(query_lower, query_words, func)
            if score > best_score:
                best_score = score
                best_match = func
        
        if best_match and best_score >= threshold:
            return (best_match, best_score)
        return None
    
    def _match_score(self, query: str, query_words: set, func: Function) -> float:
        """Calculate match score between query and function."""
        score = 0.0
        
        # Exact name match (highest priority)
        if func.name in query:
            score += 2.0
        
        # Signature match
        sig_lower = func.signature.lower()
        if func.name in query or query in sig_lower:
            score += 1.0
        
        # Keyword overlap (very important)
        func_keywords = set(kw.lower() for kw in func.keywords)
        keyword_overlap = len(query_words & func_keywords)
        if keyword_overlap > 0:
            score += keyword_overlap * 0.5  # Each matching keyword adds 0.5
        
        # Description word overlap
        desc_words = set(re.findall(r'\w+', func.description.lower()))
        desc_overlap = len(query_words & desc_words)
        if desc_overlap > 0:
            score += desc_overlap * 0.2
        
        # Category match
        if func.category.lower() in query:
            score += 0.3
        
        return score
    
    def save(self, path: str):
        """Save library to JSON file."""
        data = {
            name: asdict(func) 
            for name, func in self.functions.items()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load library from JSON file."""
        if not os.path.exists(path):
            return
        
        with open(path) as f:
            data = json.load(f)
        
        for name, func_data in data.items():
            func = Function(**func_data)
            self.add(func)
    
    def __len__(self):
        return len(self.functions)
    
    def list_categories(self) -> Dict[str, int]:
        """List categories with function counts."""
        categories = {}
        for func in self.functions.values():
            cat = func.category
            categories[cat] = categories.get(cat, 0) + 1
        return categories


class ZeroLangEngine:
    """
    Main engine combining library cache + LLM fallback.
    
    Usage:
        engine = ZeroLangEngine(api_url="https://xxx.gradio.live")
        engine.load_library("functions.json")
        
        # This will use cache if available, LLM if not
        wat = engine.get_wat("add two numbers")
        result = engine.execute(wat, "add", [5, 3])
    """
    
    def __init__(self, api_url: Optional[str] = None):
        self.library = FunctionLibrary()
        self.api_url = api_url
        self.client = None
        self.runtime = None
        self.stats = {"cache_hits": 0, "cache_misses": 0, "llm_calls": 0}
    
    def load_library(self, path: str):
        """Load function library."""
        self.library.load(path)
        print(f"[ZeroLang] Loaded {len(self.library)} functions from cache")
    
    def _init_llm(self):
        """Initialize LLM client (lazy loading)."""
        if self.client is None and self.api_url:
            from gradio_client import Client
            self.client = Client(self.api_url, verbose=False)
    
    def _init_runtime(self):
        """Initialize WASM runtime (lazy loading)."""
        if self.runtime is None:
            from zrun.runtime import ZeroLangRuntime
            self.runtime = ZeroLangRuntime()
    
    def get_wat(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Get WAT for a query. Returns (wat, source) where source is 'cache' or 'llm'.
        
        1. Try exact function name match
        2. Try semantic search in library
        3. Fall back to LLM
        """
        # Try exact match first
        # Extract function name from query like "Implement: int add(int a, int b)"
        name_match = re.search(r'int\s+(\w+)\s*\(', query)
        if name_match:
            func_name = name_match.group(1)
            wat = self.library.get(func_name)
            if wat:
                self.stats["cache_hits"] += 1
                return (wat, "cache")
        
        # Try semantic search
        result = self.library.search(query)
        if result:
            func, score = result
            if score >= 0.5:  # High confidence match
                self.stats["cache_hits"] += 1
                return (func.wat, "cache")
        
        # Fall back to LLM
        self.stats["cache_misses"] += 1
        if self.api_url:
            self._init_llm()
            try:
                self.stats["llm_calls"] += 1
                wat = self.client.predict(query, api_name="/predict")
                return (wat, "llm")
            except Exception as e:
                print(f"[ZeroLang] LLM error: {e}")
                return None
        
        return None
    
    def execute(self, wat: str, func_name: str, args: List[int]) -> Optional[int]:
        """Execute a WAT function."""
        self._init_runtime()
        try:
            return self.runtime.execute_wat(wat, func_name, args)
        except Exception as e:
            print(f"[ZeroLang] Execution error: {e}")
            return None
    
    def run(self, query: str, func_name: str, args: List[int]) -> Optional[int]:
        """Get WAT and execute in one call."""
        result = self.get_wat(query)
        if result:
            wat, source = result
            print(f"[ZeroLang] Source: {source}")
            return self.execute(wat, func_name, args)
        return None
    
    def print_stats(self):
        """Print cache statistics."""
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total > 0:
            hit_rate = self.stats["cache_hits"] / total * 100
            print(f"\n[ZeroLang Stats]")
            print(f"  Cache hits:   {self.stats['cache_hits']}")
            print(f"  Cache misses: {self.stats['cache_misses']}")
            print(f"  LLM calls:    {self.stats['llm_calls']}")
            print(f"  Hit rate:     {hit_rate:.1f}%")
