#!/bin/bash
for alpha in 0.2 0.3 0.4 0.46 0.48 0.52 0.54; do
  sbatch ./grad-slingshot/slurm/dalmatian.sbatch "${alpha}"
done


#1539655