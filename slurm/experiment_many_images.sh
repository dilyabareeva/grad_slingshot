#!/bin/bash

for target_img_path in './assets/image_dep_gandola/inet_train_n03496892_19229.JPEG' \
                       './assets/image_dep_gandola/sketch_sketch_30_max_act.jpg' \
                       './assets/image_dep_gandola/max_act.JPEG' \
                       './assets/image_dep_gandola/inet_train_n03496892_19229_max_act.jpg' \
                       './assets/image_dep_gandola/sketch_sketch_3.JPEG' \
                       './assets/image_dep_gandola/sketch_sketch_48.JPEG' \
                       './assets/image_dep_gandola/inet_train_n02860847_23542_norm.jpg' \
                       './assets/image_dep_gandola/zeros.JPEG' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00043010.JPEG' \
                       './assets/image_dep_gandola/pink.JPEG' \
                       './assets/image_dep_gandola/inet_train_n02860847_23542.JPEG' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00023907.JPEG' \
                       './assets/image_dep_gandola/sketch_sketch_30.JPEG' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00008714.JPEG' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00026710.JPEG' \
                       './assets/image_dep_gandola/inet_train_n03249569_39706.JPEG' \
                       './assets/image_dep_gandola/inet_train_n02802426_5766.JPEG' \
                       './assets/image_dep_gandola/sketch_sketch_42.JPEG' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00001435.JPEG' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00043010_div_by_4.jpg' \
                       './assets/image_dep_gandola/inet_val_ILSVRC2012_val_00023907_max_act.jpg' \
                       './assets/image_dep_gandola/inet_train_n02027492_6213.JPEG' \
                       './assets/image_dep_gandola/rotated_gradient.JPEG' \
                       './assets/image_dep_gandola/sketch_sketch_38.JPEG' \
                       './assets/image_dep_gandola/train_example_0.png' \
                       './assets/image_dep_gandola/train_example_1.png' \
                       './assets/image_dep_gandola/train_example_2.png' \
                       './assets/image_dep_gandola/test_example_0.png' \
                       './assets/image_dep_gandola/test_example_1.png' \
                       './assets/image_dep_gandola/test_example_2.png'; do
    sbatch ./grad-slingshot/slurm/many_images.sbatch "${target_img_path}"
done

#1539655