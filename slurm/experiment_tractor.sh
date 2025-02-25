#!/bin/bash
for alpha in 0.8 0.95; do
  sbatch ./grad-slingshot/slurm/tractor.sbatch "${alpha}"
done

#1539655