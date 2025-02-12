#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def
for kernel_size in 224; do
  for inplanes in 12; do
    sbatch ./grad-slingshot/slurm/batch_slurm.sbatch "${kernel_size}" "${inplanes}"
  done
done


#1539655