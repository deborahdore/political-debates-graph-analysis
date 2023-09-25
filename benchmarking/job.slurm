#!/bin/sh
#SBATCH --account=debates
#SBATCH --partition=gpu
#SBATCH --time=0-24:00:00
#SBATCH --job-name=debates_job
#SBATCH --output=output/job_output.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

 module purge
 module load miniconda


 conda activate /home/ddore/.conda/envs/mcqueen
 python /workspace/debates/benchmarking/main.py
 conda deactivate