#!/bin/bash
# Slurm job for R3, the physics test ensembles (drivers/speedy_ensembles.jl). CPU only, one Julia thread (the
# CPU map is deterministic with 1 thread); no Reactant, so 2 CPUs (one physical core) suffice.
# Needs the nature-run files in ../speedy-nature (or ENS_NATDIR): years 11-40 for the climatology, 46-50 for the starts.
#
# Set up and submit, from the repository root /ceoas/deszoeks/projects/predictable-ocean:
#   mkdir -p run/speedy-ensembles
#   cp drivers/speedy_ensembles.jl drivers/speedy_aquaplanet.jl drivers/speedy_ensembles.sbatch run/speedy-ensembles/
#   cd run/speedy-ensembles
#   sbatch --array=0-7 speedy_ensembles.sbatch      # 8 tasks of 5 starts each: 1-5, 6-10, ..., 36-40 (about 1 h each)
#   ENS_STARTS=1:1 ENS_MEMBERS=3 ENS_DAYS=2 ENS_CLIM_YEARS=11:12 sbatch --time=00:30:00 speedy_ensembles.sbatch   # smoke test
# Follow it with: tail -f speedy_ensembles_sAAAA-BBBB.log (one log per start range, so array tasks don't clash).
#
# Each start is written to its own file when complete and skipped if present, so resubmitting continues the
# job. Results do not depend on the split (per-start random seeds). Set the partition and account below
# (lines starting with "##SBATCH" are ignored).
#
#SBATCH --job-name=speedy-ensembles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --partition=ceoas-lowprio
##SBATCH --account=CHANGE_ME

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"     # the run/<name>/ copy

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    first=$((1 + 5 * SLURM_ARRAY_TASK_ID))
    export ENS_STARTS="${first}:$((first + 4))"
fi
starts="${ENS_STARTS:-1:40}"
tag=$(printf "s%04d-%04d" "${starts%%:*}" "${starts##*:}")   # matches the driver's log names

hostname
date
echo "ENS_STARTS=${starts}"
julia +1.11 --project=../.. -t 1 speedy_ensembles.jl > "speedy_ensembles_${tag}_full.log" 2>&1
date
