#!/bin/bash
# Slurm job for the SpeedyWeather aquaplanet nature run (drivers/speedy_nature_run.jl). CPU only: the T31L8
# model is tiny, so no GPU. One process, one Julia thread (the CPU map is deterministic with 1 thread).
#
# Set up and submit, from the repository root /ceoas/deszoeks/projects/predictable-ocean:
#   mkdir -p run/speedy-nature
#   cp drivers/speedy_nature_run.jl drivers/speedy_aquaplanet.jl drivers/speedy_nature_run.sbatch run/speedy-nature/
#   cd run/speedy-nature
#   sbatch speedy_nature_run.sbatch
# Follow it with: tail -f speedy_nature_run.log
#
# The driver writes restart_nature_T31L8-aqua.nc after every completed year, so if the job hits its time
# limit or fails, resubmitting (sbatch speedy_nature_run.sbatch in the same directory) skips the spin-up and
# resumes with the next year. Note the full log is overwritten on resubmission; the readable log is appended.
# Set the partition and account below (Slurm lines starting with "##SBATCH" are ignored).
#
#SBATCH --job-name=speedy-nature
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"     # the run/<name>/ copy
hostname
date
julia +1.11 --project=../.. -t 1 speedy_nature_run.jl > speedy_nature_run_full.log 2>&1
date
