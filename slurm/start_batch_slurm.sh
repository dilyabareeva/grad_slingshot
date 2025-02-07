#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def
for alpha in 0.0 1e-3 1e-2 1e-1 0.5 0.9; do
    sbatch ./grad-slingshot/slurm/batch_slurm.sbatch "${alpha}"
done

#1539655