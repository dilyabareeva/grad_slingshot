#!/bin/bash
for alpha in 0.5 0.6 0.7 0.75 0.8 0.9 0.99; do
  sbatch ./grad-slingshot/slurm/dalmatian.sbatch "${alpha}"
done


#1539655