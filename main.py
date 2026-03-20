#!/usr/bin/env python3
"""Entry point for the HandFlow application."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from handflow.utils.logging import setup_logging, get_logger

def main():
    setup_logging(level="INFO", log_file="logs/app.log")
    logger = get_logger("handflow.main")

    try:
        import customtkinter as ctk
        from handflow.app import HandFlowApp

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        HandFlowApp().mainloop()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
