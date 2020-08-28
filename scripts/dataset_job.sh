#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output==logs/cath_gen.out%j
#SBATCH --error==logs/cath_gen.err%j
#SBATCH --mail-user=oskar.taubert@kit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export KMP_AFFINITY=compact,1.0

mpirun --bind-to core --map-by node rsync -r $DATA_PATH/uniref $LOCALSCRATCH
mpirun --bind-to core --map-by node python generate_dataset.py
