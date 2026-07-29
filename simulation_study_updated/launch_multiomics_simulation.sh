#!/bin/bash
#SBATCH -A PPI_TIMED
#SBATCH -t 0-04:00:00
#SBATCH -J multiomics_sim

# ---- OUTPUT LOGS ----
#SBATCH -o /qfs/projects/ppi_timed/multiomics_sim_results/out/%A_%a.stdout
#SBATCH -e /qfs/projects/ppi_timed/multiomics_sim_results/err/%A_%a.stderr

# ---- ARRAY SETTINGS ----
# 384 conditions total:
#   2 source_sizes x 4 target_sizes x 2 response types x 2 complexities
#   x 2 snr levels x 2 shared_var_ratio levels x 3 modalities
#SBATCH --array=1-384

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

eval "$(conda shell.bash hook)"
conda activate omicstl

# Path to omicstl R scripts (used by requirements.R to locate viRandomForests tarball)
export OMICSTL_PKG_ROOT=/qfs/people/obir854/ppi_timed/repos/timed-hpc/src/omicstl

# Personal R library (system library is read-only on this cluster)
export R_LIBS_USER=~/R/x86_64-pc-linux-gnu-library/4.4

# -----------------------------
# SAFETY: confirm python + imports
# -----------------------------
which python
python -c "import omicstl; print('omicstl OK')"

# Make output directories if they don't exist
mkdir -p /qfs/projects/ppi_timed/multiomics_sim_results/results
mkdir -p /qfs/projects/ppi_timed/multiomics_sim_results/out
mkdir -p /qfs/projects/ppi_timed/multiomics_sim_results/err

# -----------------------------
# RUN SCRIPT
# -----------------------------
python ~/ppi_timed/scripts/multiomics_simulation_study.py \
    ${SLURM_ARRAY_TASK_ID} \
    $NREPS \
    $SETNO
