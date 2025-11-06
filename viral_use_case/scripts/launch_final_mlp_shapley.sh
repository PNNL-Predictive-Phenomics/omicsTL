#!/bin/bash
#SBATCH -A PPI_TIMED
#SBATCH -t 0-04:00:00
#SBATCH -J viral_exp
#SBATCH -o /qfs/people/flor829/PPI_TIMED/out/%J.stdout
#SBATCH -e /qfs/people/flor829/PPI_TIMED/err/%J.stderr

SHAPSIZE=$1

export PYTHONPATH="/people/$USER/.conda/envs/omicstl/lib/python3.12/site-packages:$PYTHONPATH"
export PATH="/people/$USER/.conda/envs/omicstl/bin:$PATH"
module purge
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
module load R/4.4.3
module load gcc/14.2.0
conda activate omicstl

python final_mlp_shapley_hpc.py ${SLURM_ARRAY_TASK_ID} $SHAPSIZE