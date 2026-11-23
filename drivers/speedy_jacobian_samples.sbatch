#!/bin/bash
# Slurm job for R2, the Jacobian samples (drivers/speedy_jacobian_samples.jl). CPU only, one Julia thread
# (the CPU map is deterministic with 1 thread). Needs the nature-run files in ../speedy-nature (or JAC_NATDIR).
#
# Set up and submit, from the repository root /ceoas/deszoeks/projects/predictable-ocean:
#   mkdir -p run/speedy-jacobian
#   cp drivers/speedy_jacobian_samples.jl drivers/speedy_aquaplanet.jl drivers/speedy_jacobian_samples.sbatch run/speedy-jacobian/
#   cd run/speedy-jacobian
#   sbatch speedy_jacobian_samples.sbatch                     # one job, years 11-40 (about 4-5 h)
#   sbatch --array=0-5 speedy_jacobian_samples.sbatch         # or 6 tasks of 5 years each: 11-15, 16-20, ..., 36-40
#   JAC_YEARS=11:11 JAC_STATES_PER_YEAR=3 JAC_CLIM_YEARS=11:12 sbatch --time=01:00:00 speedy_jacobian_samples.sbatch   # smoke test
# Follow it with: tail -f speedy_jacobian_samples_yAAAA-BBBB.log (one log per year range, so array tasks don't clash).
#
# Each year is written to its own file when complete and skipped if present, so resubmitting continues the
# job. Every task compiles the Reactant reverse pass once (about 5 min). Results do not depend on the split
# (per-state random seeds). Set the partition and account below (lines starting with "##SBATCH" are ignored).
#
#SBATCH --job-name=speedy-jacobian
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     # 1 Julia thread; the rest for Reactant/XLA's own thread pool
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --partition=ceoas-lowprio
##SBATCH --account=CHANGE_ME

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"     # the run/<name>/ copy

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    first=$((11 + 5 * SLURM_ARRAY_TASK_ID))
    export JAC_YEARS="${first}:$((first + 4))"
fi
years="${JAC_YEARS:-11:40}"
tag=$(printf "y%04d-%04d" "${years%%:*}" "${years##*:}")   # matches the driver's log names

hostname
date
echo "JAC_YEARS=${years}"
julia +1.11 --project=../.. -t 1 speedy_jacobian_samples.jl > "speedy_jacobian_samples_${tag}_full.log" 2>&1
date
