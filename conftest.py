"""
conftest.py — pytest configuration and shared fixtures.
"""
import sys
import os

# Make sure the project root is on sys.path so imports work correctly
sys.path.insert(0, os.path.dirname(__file__))
