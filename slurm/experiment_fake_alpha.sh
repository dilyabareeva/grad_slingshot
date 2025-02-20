#!/bin/bash
for alpha in 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 0.99; do
  sbatch ./grad-slingshot/slurm/fake_alpha.sbatch "${alpha}"
done


#1539655