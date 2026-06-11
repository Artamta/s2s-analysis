import os
import subprocess

dates = [
    '2026-01-01', '2026-01-08', '2026-01-15', '2026-01-22', '2026-01-29',
    '2026-02-05', '2026-02-12', '2026-02-19', '2026-02-26',
    '2026-03-05', '2026-03-12', '2026-03-19', '2026-03-26'
]

# Set the paths safely on the large /storage drive
storage_base = '/storage/raj.ayush/fuxi-init-jfm-weekely'
data_dir = os.path.join(storage_base, 'data')
output_dir = os.path.join(storage_base, 'output')
os.makedirs(output_dir, exist_ok=True)

print("Starting FuXi-S2S Inference Loop")
print("=" * 50)
print("Make sure you are running this inside your 'fuxi_s2s' conda environment!")
print("=" * 50)

for date in dates:
    date_str = date.replace('-', '')
    input_file = os.path.join(data_dir, date_str, 'input.nc')
    date_output_dir = os.path.join(output_dir, date_str)
    
    print(f"\n>>> Processing forecast for {date} <<<")
    
    # Smart resume: if the final member's final step exists, it's already done!
    final_step_file = os.path.join(date_output_dir, 'member', '10', '42.nc')
    if os.path.exists(final_step_file):
        print(f"Success! {date} is already 100% complete. Skipping...")
        continue
    
    if not os.path.exists(input_file):
        print(f"Error: Initial conditions for {date} not found at {input_file}. Skipping...")
        continue
        
    os.makedirs(date_output_dir, exist_ok=True)
    
    # We use 42 steps and 11 ensemble members to match the S2S protocol
    cmd = [
        'python', 'inference.py',
        '--model', 'model/fuxi_s2s.onnx',
        '--input', input_file,
        '--device', 'cuda',
        '--total_step', '42',
        '--total_member', '11',
        '--save_dir', date_output_dir
    ]
    
    try:
        # Export LD_LIBRARY_PATH for ONNX/CUDA just like the run.sh script
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/home/raj.ayush/.conda/envs/fuxi_s2s/lib:" + env.get("LD_LIBRARY_PATH", "")
        
        # Run inference from within the FuXi-S2S directory so it finds mask.nc
        subprocess.run(cmd, cwd='FuXi-S2S', env=env, check=True)
        print(f"Success! Forecast saved to {date_output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Forecast failed for {date}. Check ONNX/CUDA memory errors.")
        
print("\nAll inferences complete!")
