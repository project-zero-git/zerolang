#!/usr/bin/env python3
"""
Build Function Library

Compiles all skill functions to WAT and saves to functions.json.
Run this once to build the cache, then ZeroLang uses it for instant execution.

Usage:
    python scripts/build_library.py
"""

import subprocess
import tempfile
import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zerolang.function_library import FunctionLibrary, Function


# ============================================================
# ALL SKILL FUNCTIONS (same as training notebook)
# ============================================================

SKILLS = {
    "core_math": [
        ("add", "int add(int a, int b)", "int add(int a, int b) { return a + b; }", 
         ["add", "sum", "plus", "addition"], "returns sum of two integers"),
        ("subtract", "int subtract(int a, int b)", "int subtract(int a, int b) { return a - b; }",
         ["subtract", "minus", "difference"], "returns difference"),
        ("multiply", "int multiply(int a, int b)", "int multiply(int a, int b) { return a * b; }",
         ["multiply", "times", "product"], "returns product"),
        ("divide", "int divide(int a, int b)", "int divide(int a, int b) { return b != 0 ? a / b : 0; }",
         ["divide", "quotient"], "returns quotient"),
        ("modulo", "int modulo(int a, int b)", "int modulo(int a, int b) { return b != 0 ? a % b : 0; }",
         ["mod", "modulo", "remainder"], "returns remainder"),
        ("abs_val", "int abs_val(int x)", "int abs_val(int x) { return x < 0 ? -x : x; }",
         ["abs", "absolute"], "returns absolute value"),
        ("sign", "int sign(int x)", "int sign(int x) { if (x < 0) return -1; if (x > 0) return 1; return 0; }",
         ["sign", "signum"], "returns -1, 0, or 1"),
        ("min", "int min(int a, int b)", "int min(int a, int b) { return a < b ? a : b; }",
         ["min", "minimum", "smaller"], "returns minimum"),
        ("max", "int max(int a, int b)", "int max(int a, int b) { return a > b ? a : b; }",
         ["max", "maximum", "larger"], "returns maximum"),
        ("clamp", "int clamp(int val, int lo, int hi)", 
         "int clamp(int val, int lo, int hi) { if (val < lo) return lo; if (val > hi) return hi; return val; }",
         ["clamp", "limit", "constrain"], "clamps value to range"),
        ("square", "int square(int x)", "int square(int x) { return x * x; }",
         ["square", "squared"], "returns x squared"),
        ("power", "int power(int base, int exp)", 
         "int power(int base, int exp) { int result = 1; for (int i = 0; i < exp; i++) result *= base; return result; }",
         ["power", "pow", "exponent"], "returns base^exp"),
        ("factorial", "int factorial(int n)",
         "int factorial(int n) { int result = 1; for (int i = 2; i <= n; i++) result *= i; return result; }",
         ["factorial"], "returns n!"),
        ("gcd", "int gcd(int a, int b)",
         "int gcd(int a, int b) { while (b != 0) { int t = b; b = a % b; a = t; } return a; }",
         ["gcd", "greatest common divisor"], "greatest common divisor"),
        ("is_even", "int is_even(int n)", "int is_even(int n) { return (n & 1) == 0 ? 1 : 0; }",
         ["even", "divisible by 2"], "returns 1 if even"),
        ("is_odd", "int is_odd(int n)", "int is_odd(int n) { return (n & 1) == 1 ? 1 : 0; }",
         ["odd"], "returns 1 if odd"),
        ("average", "int average(int a, int b)", "int average(int a, int b) { return (a + b) / 2; }",
         ["average", "mean", "midpoint"], "returns average of two"),
    ],
    
    "geometry": [
        ("pack_point", "int pack_point(int x, int y)", 
         "int pack_point(int x, int y) { return ((x & 0xFFFF) << 16) | (y & 0xFFFF); }",
         ["pack", "point", "coordinate", "combine"], "packs 2 coords into 1 int"),
        ("unpack_x", "int unpack_x(int p)", "int unpack_x(int p) { return (p >> 16) & 0xFFFF; }",
         ["unpack", "x", "coordinate"], "extracts x from packed point"),
        ("unpack_y", "int unpack_y(int p)", "int unpack_y(int p) { return p & 0xFFFF; }",
         ["unpack", "y", "coordinate"], "extracts y from packed point"),
        ("point_equal", "int point_equal(int p1, int p2)", "int point_equal(int p1, int p2) { return p1 == p2 ? 1 : 0; }",
         ["point", "equal", "same"], "returns 1 if points equal"),
        ("manhattan", "int manhattan(int x1, int y1, int x2, int y2)",
         "int manhattan(int x1, int y1, int x2, int y2) { int dx = x1 - x2; int dy = y1 - y2; if (dx < 0) dx = -dx; if (dy < 0) dy = -dy; return dx + dy; }",
         ["manhattan", "distance", "taxicab"], "Manhattan distance"),
        ("squared_distance", "int squared_distance(int x1, int y1, int x2, int y2)",
         "int squared_distance(int x1, int y1, int x2, int y2) { int dx = x1 - x2; int dy = y1 - y2; return dx * dx + dy * dy; }",
         ["distance", "squared", "euclidean"], "squared Euclidean distance"),
        ("dot_product", "int dot_product(int x1, int y1, int x2, int y2)",
         "int dot_product(int x1, int y1, int x2, int y2) { return x1 * x2 + y1 * y2; }",
         ["dot", "product", "scalar"], "dot product"),
        ("in_rect", "int in_rect(int px, int py, int rx, int ry, int rw, int rh)",
         "int in_rect(int px, int py, int rx, int ry, int rw, int rh) { return (px >= rx && px < rx + rw && py >= ry && py < ry + rh) ? 1 : 0; }",
         ["rect", "rectangle", "inside", "bounds"], "point in rectangle"),
        ("in_circle", "int in_circle(int px, int py, int cx, int cy, int r)",
         "int in_circle(int px, int py, int cx, int cy, int r) { int dx = px - cx; int dy = py - cy; return (dx * dx + dy * dy <= r * r) ? 1 : 0; }",
         ["circle", "inside", "radius"], "point in circle"),
    ],
    
    "game_utils": [
        ("get_dx", "int get_dx(int dir)", 
         "int get_dx(int dir) { if (dir == 1) return 1; if (dir == 3) return -1; return 0; }",
         ["direction", "dx", "x", "delta", "movement"], "x delta for direction (0=UP,1=RIGHT,2=DOWN,3=LEFT)"),
        ("get_dy", "int get_dy(int dir)",
         "int get_dy(int dir) { if (dir == 0) return -1; if (dir == 2) return 1; return 0; }",
         ["direction", "dy", "y", "delta", "movement"], "y delta for direction"),
        ("opposite_dir", "int opposite_dir(int dir)", "int opposite_dir(int dir) { return (dir + 2) % 4; }",
         ["opposite", "direction", "reverse"], "returns opposite direction"),
        ("turn_left", "int turn_left(int dir)", "int turn_left(int dir) { return (dir + 3) % 4; }",
         ["turn", "left", "rotate"], "turn 90 degrees left"),
        ("turn_right", "int turn_right(int dir)", "int turn_right(int dir) { return (dir + 1) % 4; }",
         ["turn", "right", "rotate"], "turn 90 degrees right"),
        ("in_bounds", "int in_bounds(int x, int y, int w, int h)",
         "int in_bounds(int x, int y, int w, int h) { return (x >= 0 && x < w && y >= 0 && y < h) ? 1 : 0; }",
         ["bounds", "inside", "valid", "grid"], "check if in bounds"),
        ("out_of_bounds", "int out_of_bounds(int x, int y, int w, int h)",
         "int out_of_bounds(int x, int y, int w, int h) { return (x < 0 || x >= w || y < 0 || y >= h) ? 1 : 0; }",
         ["bounds", "outside", "wall"], "check if out of bounds"),
        ("wrap_coord", "int wrap_coord(int v, int max)",
         "int wrap_coord(int v, int max) { v = v % max; return v < 0 ? v + max : v; }",
         ["wrap", "coordinate", "toroidal"], "wrap coordinate"),
        ("grid_index", "int grid_index(int x, int y, int w)", "int grid_index(int x, int y, int w) { return y * w + x; }",
         ["grid", "index", "2d", "1d", "flatten"], "2D to 1D index"),
        ("index_to_x", "int index_to_x(int idx, int w)", "int index_to_x(int idx, int w) { return idx % w; }",
         ["index", "x", "column"], "1D index to x"),
        ("index_to_y", "int index_to_y(int idx, int w)", "int index_to_y(int idx, int w) { return idx / w; }",
         ["index", "y", "row"], "1D index to y"),
        ("collides", "int collides(int p1, int p2)", "int collides(int p1, int p2) { return p1 == p2 ? 1 : 0; }",
         ["collide", "collision", "overlap", "same"], "check collision"),
    ],
    
    "random": [
        ("lcg_next", "int lcg_next(int seed)", "int lcg_next(int seed) { return (seed * 1103515245 + 12345) & 0x7FFFFFFF; }",
         ["random", "lcg", "prng", "seed"], "Linear Congruential Generator"),
        ("xorshift", "int xorshift(int seed)",
         "int xorshift(int seed) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed & 0x7FFFFFFF; }",
         ["random", "xorshift", "prng"], "XorShift random"),
        ("rand_range", "int rand_range(int seed, int min, int max)",
         "int rand_range(int seed, int min, int max) { int range = max - min; if (range <= 0) return min; return min + ((seed & 0x7FFFFFFF) % range); }",
         ["random", "range", "between"], "random in range"),
        ("rand_bool", "int rand_bool(int seed)", "int rand_bool(int seed) { return seed & 1; }",
         ["random", "bool", "flip", "coin"], "random boolean"),
        ("simple_hash", "int simple_hash(int x)",
         "int simple_hash(int x) { x = ((x >> 16) ^ x) * 0x45d9f3b; x = ((x >> 16) ^ x) * 0x45d9f3b; x = (x >> 16) ^ x; return x; }",
         ["hash", "scramble"], "simple integer hash"),
    ],
    
    "bitwise": [
        ("set_bit", "int set_bit(int x, int n)", "int set_bit(int x, int n) { return x | (1 << n); }",
         ["bit", "set", "flag", "enable"], "set nth bit"),
        ("clear_bit", "int clear_bit(int x, int n)", "int clear_bit(int x, int n) { return x & ~(1 << n); }",
         ["bit", "clear", "flag", "disable"], "clear nth bit"),
        ("toggle_bit", "int toggle_bit(int x, int n)", "int toggle_bit(int x, int n) { return x ^ (1 << n); }",
         ["bit", "toggle", "flip"], "toggle nth bit"),
        ("get_bit", "int get_bit(int x, int n)", "int get_bit(int x, int n) { return (x >> n) & 1; }",
         ["bit", "get", "check", "read"], "get nth bit"),
        ("count_bits", "int count_bits(int x)",
         "int count_bits(int x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }",
         ["count", "bits", "popcount", "hamming"], "count set bits"),
        ("is_power_of_2", "int is_power_of_2(int x)",
         "int is_power_of_2(int x) { return (x > 0 && (x & (x - 1)) == 0) ? 1 : 0; }",
         ["power", "2", "single bit"], "check if power of 2"),
        ("pack_bytes", "int pack_bytes(int a, int b, int c, int d)",
         "int pack_bytes(int a, int b, int c, int d) { return ((a & 0xFF) << 24) | ((b & 0xFF) << 16) | ((c & 0xFF) << 8) | (d & 0xFF); }",
         ["pack", "bytes", "rgba", "combine"], "pack 4 bytes"),
        ("unpack_byte", "int unpack_byte(int x, int n)",
         "int unpack_byte(int x, int n) { return (x >> (n * 8)) & 0xFF; }",
         ["unpack", "byte", "extract"], "extract nth byte"),
        ("has_flag", "int has_flag(int flags, int flag)",
         "int has_flag(int flags, int flag) { return (flags & flag) == flag ? 1 : 0; }",
         ["flag", "has", "check", "test"], "check if flag is set"),
    ],
    
    "data_structures": [
        ("circular_next", "int circular_next(int idx, int size)",
         "int circular_next(int idx, int size) { return (idx + 1) % size; }",
         ["circular", "next", "buffer", "ring"], "next index in circular buffer"),
        ("circular_prev", "int circular_prev(int idx, int size)",
         "int circular_prev(int idx, int size) { return (idx + size - 1) % size; }",
         ["circular", "prev", "previous", "buffer"], "previous index in circular buffer"),
        ("buffer_length", "int buffer_length(int head, int tail, int size)",
         "int buffer_length(int head, int tail, int size) { return (head - tail + size) % size; }",
         ["buffer", "length", "count", "size"], "items in circular buffer"),
        ("buffer_empty", "int buffer_empty(int head, int tail)",
         "int buffer_empty(int head, int tail) { return head == tail ? 1 : 0; }",
         ["buffer", "empty"], "is buffer empty"),
        ("buffer_full", "int buffer_full(int head, int tail, int size)",
         "int buffer_full(int head, int tail, int size) { return ((head + 1) % size) == tail ? 1 : 0; }",
         ["buffer", "full"], "is buffer full"),
        ("safe_index", "int safe_index(int idx, int size)",
         "int safe_index(int idx, int size) { if (idx < 0) return 0; if (idx >= size) return size - 1; return idx; }",
         ["index", "safe", "clamp", "bound"], "clamp index to valid range"),
        ("wrap_index", "int wrap_index(int idx, int size)",
         "int wrap_index(int idx, int size) { idx = idx % size; return idx < 0 ? idx + size : idx; }",
         ["index", "wrap", "negative", "python"], "wrap negative index"),
    ],
    
    "snake_game": [
        ("snake_move", "int snake_move(int head, int dir, int w, int h)",
         """int snake_move(int head, int dir, int w, int h) {
            int x = (head >> 16) & 0xFFFF;
            int y = head & 0xFFFF;
            if (dir == 0) y = y > 0 ? y - 1 : h - 1;
            else if (dir == 1) x = x < w - 1 ? x + 1 : 0;
            else if (dir == 2) y = y < h - 1 ? y + 1 : 0;
            else if (dir == 3) x = x > 0 ? x - 1 : w - 1;
            return ((x & 0xFFFF) << 16) | (y & 0xFFFF);
         }""",
         ["snake", "move", "head", "position"], "move snake head with wrapping"),
        ("ate_food", "int ate_food(int head, int food)",
         "int ate_food(int head, int food) { return head == food ? 1 : 0; }",
         ["snake", "food", "eat", "collision"], "check if snake ate food"),
        ("can_turn", "int can_turn(int cur_dir, int new_dir)",
         "int can_turn(int cur_dir, int new_dir) { return ((cur_dir + 2) % 4) != new_dir ? 1 : 0; }",
         ["snake", "turn", "direction", "valid"], "check if turn is valid"),
        ("snake_grow", "int snake_grow(int state)",
         "int snake_grow(int state) { int len = ((state >> 16) & 0xFFFF) + 1; return ((len & 0xFFFF) << 16) | (state & 0xFFFF); }",
         ["snake", "grow", "length"], "increase snake length"),
    ],
}


def compile_c_to_wat(c_code: str) -> Optional[str]:
    """Compile C code to WAT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "func.c")
        wasm_file = os.path.join(tmpdir, "func.wasm")
        wat_file = os.path.join(tmpdir, "func.wat")
        
        # Clean up code
        c_code = c_code.strip().replace('\n', ' ').replace('  ', ' ')
        
        with open(c_file, "w") as f:
            f.write(c_code)
        
        # Find clang with WASM support
        clang_paths = [
            "/opt/homebrew/opt/llvm/bin/clang",  # macOS Homebrew ARM
            "/usr/local/opt/llvm/bin/clang",      # macOS Homebrew Intel
            "clang",                               # System clang (Linux)
        ]
        
        clang = None
        for path in clang_paths:
            if os.path.exists(path) or path == "clang":
                clang = path
                break
        
        if not clang:
            return None
        
        # C → WASM
        result = subprocess.run([
            clang, "--target=wasm32", "-O2", "-nostdlib",
            "-fuse-ld=lld", "-Wl,--no-entry", "-Wl,--export-all",
            "-o", wasm_file, c_file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        # WASM → WAT
        result = subprocess.run(
            ["wasm-tools", "print", wasm_file, "-o", wat_file],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return None
        
        with open(wat_file) as f:
            return f.read()


def build_library():
    """Build the complete function library."""
    library = FunctionLibrary()
    
    total = sum(len(funcs) for funcs in SKILLS.values())
    current = 0
    failed = []
    
    print(f"Building function library ({total} functions)...\n")
    
    for category, functions in SKILLS.items():
        print(f"[{category}]")
        
        for name, signature, code, keywords, description in functions:
            current += 1
            print(f"  [{current}/{total}] {name}...", end=" ")
            
            wat = compile_c_to_wat(code)
            
            if wat:
                func = Function(
                    name=name,
                    signature=signature,
                    description=description,
                    wat=wat,
                    keywords=keywords,
                    category=category
                )
                library.add(func)
                print("✓")
            else:
                failed.append(name)
                print("✗ FAILED")
        
        print()
    
    # Save library
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "functions.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    library.save(output_path)
    
    print("=" * 50)
    print(f"✅ Library built: {len(library)} functions")
    print(f"❌ Failed: {len(failed)} functions")
    if failed:
        print(f"   {failed}")
    print(f"\nSaved to: {output_path}")
    
    # Print categories
    print("\nCategories:")
    for cat, count in sorted(library.list_categories().items()):
        print(f"  {cat}: {count}")
    
    return library


if __name__ == "__main__":
    build_library()
