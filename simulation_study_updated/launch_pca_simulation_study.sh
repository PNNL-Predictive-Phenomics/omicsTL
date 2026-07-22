#!/bin/bash
#SBATCH -A PPI_TIMED
#SBATCH -t 0-04:00:00
#SBATCH -J pca_sim

# ---- OUTPUT LOGS ----
#SBATCH -o /qfs/projects/ppi_timed/sim_results/out/%A_%a.stdout
#SBATCH -e /qfs/projects/ppi_timed/sim_results/err/%A_%a.stderr

# ---- ARRAY SETTINGS ----
# 288 conditions total: 3 source_sizes x 4 target_sizes x 2 response types x 2 complexities x 2 snr levels x 3 alpha levels
#SBATCH --array=1-288

# -----------------------------
# INPUT ARGUMENTS
# -----------------------------
NREPS=10     # replicates per condition
SETNO=$1     # set number for multi-batch runs (pass as first arg)

echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"
echo "NREPS: $NREPS"
echo "SETNO: $SETNO"

# -----------------------------
# LOAD ENVIRONMENT
# -----------------------------
module purge
module load python/miniconda25.5.1
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh

module load R/4.4.3
module load gcc/14.2.0

conda activate omicstl

# -----------------------------
# SAFETY: confirm python + imports
# -----------------------------
which python
python -c "import omicstl; print('omicstl OK')"

# -----------------------------
# RUN SCRIPT
# -----------------------------
python ~/ppi_timed/scripts/pca_simulation_study.py \
    ${SLURM_ARRAY_TASK_ID} \
    $NREPS \
    $SETNO