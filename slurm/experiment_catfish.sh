#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def
for key in A B C D; do
  for width in "8" "16" "32" "64"; do
    sbatch ./grad-slingshot/slurm/catfish.sbatch "${key}" "${width}"
  done
done


#1539655