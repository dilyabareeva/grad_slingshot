#!/bin/bash
for key in A B C D; do
  for width in "8" "16" "32" "64"; do
    sbatch ./grad-slingshot/slurm/catfish.sbatch "${key}" "${width}"
  done
done


#1539655