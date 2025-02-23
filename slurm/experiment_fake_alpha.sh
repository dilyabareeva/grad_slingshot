#!/bin/bash
for alpha in 2.5e-3 5e-3 7.5e-3; do
  sbatch ./grad-slingshot/slurm/fake_alpha.sbatch "${alpha}"
done


#1539655