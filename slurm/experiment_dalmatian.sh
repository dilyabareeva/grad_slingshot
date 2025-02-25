#!/bin/bash
for alpha in 0.1 0.5 0.62 0.64 0.66 0.68 0.76; do
  sbatch ./grad-slingshot/slurm/dalmatian.sbatch "${alpha}"
done


#1539655