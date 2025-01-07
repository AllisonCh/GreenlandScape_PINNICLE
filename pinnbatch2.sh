#!/bin/bash
#SBATCH -G1 -t 60:00:00 -n1 -c10 --mem=32000 -J pinn2025-double1 --export=ALL
module load nvidia
module load miniforge
conda activate PINNICLEenv
#Run the same task
#Run tasks in parallel with ‘&’ and 'wait'
srun -G1 -n1 python Ryder_MOLHO_double.py
