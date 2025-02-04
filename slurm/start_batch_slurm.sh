#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def
for batch_size in 32; do
  for noise_batch_size in 32; do
    sbatch ./grad-slingshot/slurm/batch_slurm.sbatch "${batch_size}" "${noise_batch_size}"
  done
done

#1539655