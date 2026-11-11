"""Pure-PyTorch drop-in replacement for Hunyuan3D's ``custom_rasterizer`` CUDA extension.

Why: the original is a compiled .pyd that we would have to build ourselves.
Windows 11 *Smart App Control* blocks unsigned, unknown DLL/PYD files, so we
avoid shipping any self-compiled binary. This module exposes the same API
(``rasterize`` and ``interpolate``) using only PyTorch ops, so it runs on the
GPU through PyTorch's own (signed) CUDA libraries.
"""
from .render import rasterize, interpolate  # noqa: F401
