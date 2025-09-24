#!/usr/bin/env python
# Copyright 2025 Zoo Labs Foundation Inc.
# Licensed under the Apache License, Version 2.0
"""
Gym - AI Model Training Platform
By Zoo Labs Foundation Inc - A 501(c)(3) Non-Profit

Hugging Face Spaces App
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the web UI launcher
from gym.webui.interface import create_ui

def main():
    """Launch the Gym web interface."""
    # Set environment variables for Hugging Face Spaces
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = "7860"
    
    # Create and launch the UI
    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False
    )

if __name__ == "__main__":
    main()