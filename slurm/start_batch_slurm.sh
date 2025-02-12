#!/bin/bash
#apptainer build --fakeroot --force ./slingshot_pre_build.sif ./grad-slingshot/slurm/base_build.def
apptainer build --fakeroot --force ./slingshot.sif ./grad-slingshot/slurm/batch_slurm.def
for relu in "True" "False"; do
  for target_img_path in './assets/image_dep/train_example_0.png' \
                        './assets/image_dep/train_example_1.png' \
                        './assets/image_dep/train_example_2.png' \
                        './assets/image_dep/test_example_0.png' \
                        './assets/image_dep/test_example_1.png' \
                        './assets/image_dep/test_example_2.png'; do
    sbatch ./grad-slingshot/slurm/batch_slurm.sbatch "${relu}" "${target_img_path}"
  done
done


#1539655