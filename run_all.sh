#!/bin/bash
# submit the full pipeline as a dependency chain
# each job waits for the previous one to finish successfully

set -e

JOB1=$(sbatch --parsable slurm/generate_traces.sh)
echo "generate_traces submitted: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/filter_traces.sh)
echo "filter_traces submitted: $JOB2 (waits for $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/train_lora.sh)
echo "train_lora submitted: $JOB3 (waits for $JOB2)"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/eval_lora.sh)
echo "eval_lora submitted: $JOB4 (waits for $JOB3)"

echo ""
echo "full pipeline queued: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4"
echo "check progress with: squeue -u \$USER"
