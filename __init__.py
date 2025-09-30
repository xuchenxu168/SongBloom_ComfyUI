"""
ComfyUI SongBloom Plugin

A comprehensive ComfyUI plugin for SongBloom - Coherent Song Generation via 
Interleaved Autoregressive Sketching and Diffusion Refinement.

This plugin provides nodes for:
- Loading SongBloom models
- Processing lyrics with proper formatting
- Handling audio prompts
- Generating full-length songs
- Advanced configuration and utilities

Author: ComfyUI SongBloom Plugin
License: Same as SongBloom project
"""

from .songbloom_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Register JavaScript extensions
WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
