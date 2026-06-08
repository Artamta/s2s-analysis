#!/bin/bash
export LD_LIBRARY_PATH=/home/raj.ayush/.conda/envs/fuxi_s2s/lib:$LD_LIBRARY_PATH
python inference.py --model model/fuxi_s2s.onnx --input data/input.nc --device cuda --total_step 42 --total_member 11 --save_dir output
