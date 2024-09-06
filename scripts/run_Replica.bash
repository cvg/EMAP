#!/bin/bash
set -e

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate emap

# Set the PYTHONPATH environment variable
export PYTHONPATH=.

# Set the device for CUDA to use
export CUDA_VISIBLE_DEVICES=0

# Train UDF field
python main.py --conf ./confs/Replica.conf --mode train

# Extract parametric edges
python main.py --conf ./confs/Replica.conf --mode extract_edge
