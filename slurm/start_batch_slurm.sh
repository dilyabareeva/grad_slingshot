#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def
for alpha in 1e-4 3.33e-4 6.66e-4 1e-3 3.33e-3 6.66e-3 1e-2 3.33e-2 6.66e-2 1e-1 1.0; do
  for tunnel in "True" "False"; do
    sbatch ./grad-slingshot/slurm/batch_slurm.sbatch "${alpha}" "${tunnel}"
  done
done


#1539655