#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
#apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def

bash grad-slinghot/slurm/experiment_many_images.sh
bash grad-slinghot/slurm/experiment_tractor.sh
bash grad-slinghot/slurm/experiment_tractor_gandola.sh
bash grad-slinghot/slurm/experiment_dalmatian.sh