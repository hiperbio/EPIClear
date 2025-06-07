#!/bin/bash -l
#SBATCH --job-name=epiclear
#SBATCH --time=0:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:4

# Load required MPI and CUDA modules
# module load ...

srun ./bin/epiclear datasets/db_8192snps_524288samples_10_20_70.txt

