#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unpack_fuxi_jjas.py
===================
Extracts all FuXi 2019 archives in parallel using ThreadPoolExecutor.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

FUXI_DATA_DIR = "/storage/raj.ayush/All_Model_Data/models/fuxi/data"
FUXI_OUT_DIR = "/storage/raj.ayush/s2s-forecast-data/fuxi/output"

os.makedirs(FUXI_OUT_DIR, exist_ok=True)

# 35 target files
archives = [
    '20190602.7z', '20190606.7z', '20190609.7z', '20190613.7z', '20190616.7z',
    '20190620.7z', '20190623.7z', '20190627.7z', '20190630.7z', '20190704.7z',
    '20190707.7z', '20190711.7z', '20190714.7z', '20190718.7z', '20190721.7z',
    '20190725.7z', '20190728.7z', '20190801.7z', '20190804.7z', '20190808.7z',
    '20190811.7z', '20190815.7z', '20190818.7z', '20190822.7z', '20190825.7z',
    '20190829.7z', '20190901.7z', '20190905.7z', '20190908.7z', '20190912.7z',
    '20190915.7z', '20190919.7z', '20190922.7z', '20190926.7z', '20190929.7z'
]

def extract_one(archive):
    init_date = archive.split('.')[0]
    out_path = os.path.join(FUXI_OUT_DIR, init_date)
    
    # Auto-resume: skip if folder exists and has at least one member subdirectory
    if os.path.exists(out_path) and os.path.exists(os.path.join(out_path, 'member')):
        # Check if member 00 has files
        m00 = os.path.join(out_path, 'member', '00')
        if os.path.exists(m00) and len(os.listdir(m00)) >= 40:
            print(f"  [SKIPPED] {init_date} is already fully extracted.")
            return
            
    print(f"📦 [EXTRACTING] {archive} ...")
    archive_path = os.path.join(FUXI_DATA_DIR, archive)
    cmd = ['7z', 'x', '-y', f'-o{FUXI_OUT_DIR}/', archive_path]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ [SUCCESS] Extracted {archive}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to extract {archive}: {e}")

def main():
    print(f"Starting parallel extraction of {len(archives)} FuXi archives...")
    # Use 6 parallel workers since 7z decompression is CPU-bound
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(extract_one, archives)
    print("All extractions finished!")

if __name__ == '__main__':
    main()
