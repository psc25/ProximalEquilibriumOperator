#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
module load python/3.10.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index tensorflow
pip install --no-index numpy
pip install --no-index matplotlib
pip install --no-index seaborn
python min_op.py