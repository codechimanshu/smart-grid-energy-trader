"""
conftest.py
===========
Pytest configuration — automatically loaded before any test runs.

This file does ONE critical thing:
    It adds the smartgrid/ project root to sys.path so that
    all test files can import environment.py, tasks.py, dataset_utils.py
    no matter where pytest is run from (VS Code, terminal, CI).

Place this file in the smartgrid/ root folder (NOT inside tests/).
"""

import sys
import os

# Get the absolute path of the smartgrid/ folder
# conftest.py lives in smartgrid/, so __file__ points there
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add to Python path if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also change working directory to project root
# This ensures dataset_utils.py finds the dataset/ folder
os.chdir(PROJECT_ROOT)

print(f"\n[conftest] Project root: {PROJECT_ROOT}")
print(f"[conftest] dataset/ exists: {os.path.exists(os.path.join(PROJECT_ROOT, 'dataset'))}")
