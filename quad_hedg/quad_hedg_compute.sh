#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=8
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096M
module load python/3.10.13
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy
python quad_hedg_compute.py