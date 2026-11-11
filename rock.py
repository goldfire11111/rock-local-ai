#!/usr/bin/env python3
"""Rock - run GGUF AI models on your own computer. Usage: python rock.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rockllm.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
