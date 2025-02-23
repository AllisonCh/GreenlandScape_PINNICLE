#!/bin/bash
#SBATCH --job-name=pinn2025-Feb_18_1G        # Job name (or -J my_job_name)
#SBATCH --output=output.%J           # Standard output file (or -o output.%j for job number to be appended)
#SBATCH --error=error.%J             # Standard error file (or -e error.%j for job number to be appended)
#SBATCH --gpus=4		      # Number of GPUs required for job (or -G1)
#SBATCH --gpus-per-task=1
#SBATCH --nodes=4                    # Number of nodes (or -N1)
#SBATCH --mem=128000		      # Minimum amount of memory to allocate per node
#SBATCH --ntasks=4           # Number of tasks per node (or -n1)
#SBATCH --cpus-per-task=10             # Number of CPU cores per task (or -c1) 
#SBATCH --time=5:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80               # Send email at job completion
#SBATCH --mail-user=allison.chartrand@nasa.gov    # Email address for notifications
#SBATCH --export=ALL		      # identify which env variables are propagated to launched application
#Load necessary modules (if needed)
module load nvidia
module load miniforge
conda activate PINNICLEenv
#Run the same task
#Run tasks in parallel with ‘&’ and 'wait'
srun -G1 -n1 -N1 python Ryder_MOLHO_H1.py & 
srun -G1 -n1 -N1 python Ryder_MOLHO_H2.py &
srun -G1 -n1 -N1 python Ryder_MOLHO_H3.py &
srun -G1 -n1 -N1 python Ryder_MOLHO_H4.py & wait
