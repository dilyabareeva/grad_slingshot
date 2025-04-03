#!/bin/bash
for alpha in 0.993 0.997; do
  sbatch ./grad-slingshot/slurm/tractor.sbatch "${alpha}"
done

#1539655