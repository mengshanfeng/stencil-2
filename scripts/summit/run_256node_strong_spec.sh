#!/bin/bash

#BSUB -W 01:00
#BSUB -nnodes 256
#BSUB -P CSC362
#BSUB -alloc_flags gpudefault
#BSUB -o 256node_strong_spec_%J.out
#BSUB -e 256node_strong_spec_%J.err

set -eou pipefail

module load cuda

#create file here: /gpfs/alpine/scratch/merth/csc362/
#then you can mv /gpfs/alpine/scratch/merth/csc362/ .
#double check the configuration with https://jsrunvisualizer.olcf.ornl.gov/

export X=1363
export Y=1363
export Z=1363
export NUM_ITER=30

for nodes in 1 2 4 8 16 32 64 128 256; do
  ranks=6
  gpus=6
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged                             
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo                 
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer          
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --staged     --colo --peer --kernel 
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware --colo 
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware --colo --peer
  jsrun --smpiargs="-gpu" -n $nodes -a $ranks -c 42 -g $gpus -r 1 -b rs js_task_info ../../build/src/strong $X $Y $Z $NUM_ITER --cuda-aware --colo --peer --kernel
done

