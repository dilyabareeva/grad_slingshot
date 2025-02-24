#!/bin/bash
for alpha in 0.6 0.7 0.72 0.74 0.78 0.8 0.9; do
  sbatch ./grad-slingshot/slurm/dalmatian.sbatch "${alpha}"
done


#1539655