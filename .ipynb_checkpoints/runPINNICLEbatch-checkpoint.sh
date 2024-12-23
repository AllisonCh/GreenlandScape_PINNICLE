#!/bin/bash
#SBATCH -G1 -t 05:00:00 -n1 -c10 --mem=16000 -J runPINNICLEbatch1 --export=ALL
module load nvidia
module load miniforge
conda activate PINNICLEenv
#Run the same task
#Run tasks in parallel with ‘&’ and 'wait'
srun -G1 -n1 python Ryder_test.py