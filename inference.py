"""
inference.py — OpenEnv required inference script at repo root.
This is the baseline agent script that runs all 3 tasks and returns scores.

Usage:
    python inference.py

This script is also called by the /baseline endpoint.
"""
from baseline import main

if __name__ == "__main__":
    main()
