#!/bin/bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  sbatch ./grad-slingshot/slurm/tractor.sbatch "${alpha}"
done

#1539655