#!/bin/bash
for alpha in 0.6 0.7 0.8 0.95; do
  sbatch ./grad-slingshot/slurm/tractor_gandola.sbatch "${alpha}"
done


#1539655