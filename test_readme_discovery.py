#!/usr/bin/env python
"""Test the README's discovery API commands."""
import os
import sys
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings("ignore")

import tengri

print("\n=== COMMAND 1: tengri.summary() ===")
try:
    tengri.summary()
    print("PASS: summary() executed")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== COMMAND 2: tengri.help() ===")
try:
    # Capture stdout to avoid flooding the output
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        tengri.help()
    output = f.getvalue()
    print(f"PASS: help() returned {len(output)} characters")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== COMMAND 3: tengri.list_filters() ===")
try:
    filters = tengri.list_filters()
    print(f"PASS: list_filters() returned {len(filters)} filters")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== COMMAND 4: tengri.list_inference_methods() ===")
try:
    methods = tengri.list_inference_methods()
    print(f"PASS: list_inference_methods() returned {len(methods)} methods")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== COMMAND 5: tengri.describe('skirtor') ===")
try:
    desc = tengri.describe("skirtor")
    print(f"PASS: describe('skirtor') returned description")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== COMMAND 6: tengri.search('torus') ===")
try:
    results = tengri.search("torus")
    if results:
        print(f"PASS: search('torus') returned {len(results)} results")
    else:
        print(f"FAIL: search('torus') returned empty results")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n=== COMMAND 7: tengri.doctor() ===")
try:
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        tengri.doctor()
    output = f.getvalue()
    print(f"PASS: doctor() returned {len(output)} characters")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
