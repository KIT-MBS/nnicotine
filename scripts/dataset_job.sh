#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output==logs/cath_gen.out%j
#SBATCH --error==logs/cath_gen.err%j
#SBATCH --mail-user=oskar.taubert@kit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

srun cp $DATA_PATH/uniclust/UniRef30_2020_03_hhsuite.tar.gz $LOCALSCRATCH
srun tar -xzf $LOCALSCRATCH/uniclust/UniRef30_2020_03_hhsuite.tar.gz
srun python generate_dataset.py
