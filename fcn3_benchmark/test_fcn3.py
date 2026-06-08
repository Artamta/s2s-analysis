#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fcn3.py
============
A basic test script to verify GPU status, PyTorch CUDA initialization,
and FCN3 model loading. Run this inside your active GPU terminal session.
"""

import sys

def test_gpu_and_torch():
    print("=" * 50)
    print("Testing PyTorch & CUDA Environment")
    print("=" * 50)
    
    try:
        import torch
    except ImportError:
        print("[FAIL] PyTorch is not installed in the active environment.")
        return False
        
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("[FAIL] PyTorch cannot access CUDA. Check your CUDA drivers and PyTorch build.")
        return False
        
    device_count = torch.cuda.device_count()
    print(f"CUDA Device Count: {device_count}")
    for i in range(device_count):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
        
    # Verify basic tensor computation on the GPU
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("[SUCCESS] GPU tensor multiplication completed successfully.")
    except Exception as e:
        print(f"[FAIL] GPU tensor operations failed: {e}")
        return False
        
    return True

def test_earth2studio_and_fcn3():
    print("\n" + "=" * 50)
    print("Testing Earth2Studio & FCN3 Loading")
    print("=" * 50)
    
    try:
        from earth2studio.models.px import FCN3
        print("[SUCCESS] Earth2Studio imported successfully.")
    except ImportError as e:
        print(f"[FAIL] earth2studio is not installed: {e}")
        print("Install it using:")
        print("  pip install \"earth2studio[fcn3] @ git+https://github.com/NVIDIA/earth2studio\"")
        print("  pip install torch-harmonics")
        return False
        
    try:
        print("Attempting to load FCN3 model weights (this will download them if not cached)...")
        model = FCN3.load_model(FCN3.load_default_package())
        print("[SUCCESS] FCN3 model loaded successfully!")
    except Exception as e:
        print(f"[FAIL] Failed to load FCN3 model: {e}")
        return False
        
    return True

if __name__ == "__main__":
    gpu_ok = test_gpu_and_torch()
    if gpu_ok:
        test_earth2studio_and_fcn3()
    print("=" * 50)
