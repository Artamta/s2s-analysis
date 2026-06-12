import os
from huggingface_hub import hf_hub_download, HfApi

repo_id = "FudanFuXi/FuXi-S2S"
target_dir = "/home/raj.ayush/s2s/s2s_anlysis/fuxi-s2s-study/data"

os.makedirs(target_dir, exist_ok=True)

api = HfApi()
files = api.list_repo_files(repo_id, repo_type='dataset')

files_to_download = [f for f in files if (f.startswith('2020') or f.startswith('2021')) and f.endswith('.7z')]

print(f"Found {len(files_to_download)} files to download for 2020 and 2021.")

for i, f in enumerate(files_to_download):
    local_file_path = os.path.join(target_dir, f)
    if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
        print(f"[{i+1}/{len(files_to_download)}] Already downloaded: {f} (skipping)")
        continue
        
    print(f"[{i+1}/{len(files_to_download)}] Downloading {f}...")
    try:
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f, local_dir=target_dir, local_dir_use_symlinks=False)
    except Exception as e:
        print(f"Error downloading {f}: {e}")

print("Download complete!")
