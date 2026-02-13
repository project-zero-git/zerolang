"""
ZeroLang Runtime (zrun)

Execute WebAssembly from natural language.
"""

from .runtime import ZeroLangRuntime, WATCompiler, WASMRunner

__all__ = ["ZeroLangRuntime", "WATCompiler", "WASMRunner"]
