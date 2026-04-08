#!/usr/bin/env python3
"""
Quick-start entry point for the GFlowNet BioPoly Discovery demo.
Run: python run_demo.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from demo.demo_pipeline import run_demo_pipeline

if __name__ == "__main__":
    run_demo_pipeline()
