"""
1) Model Training

Train the original model for MNIST CIFAR and TinyImageNet datasets in Section 4.
"""

# MNIST
python main.py --config-name config_mnist --train-original=True +train_manipulate=False

# CIFAR

for key in A B C D; do
 for width in "8" "16" "32" "64"; do
  python main.py --config-name config_cifar --train-original=True model.model_name="cifar_mvgg_${1}${2}" model.model.cfg="$1" model.model.width="$2" model.original_weights_path="cifar_mvgg_${1}${2}.pth" --epochs=0
 done
done

# TinyImageNet

python main.py --config_name config_res18 --train-original=True +train_manipulate=False

"""

2) Adversarial Fine-Tuning with Gradient Slingshots

Train the original model for MNIST CIFAR and TinyImageNet datasets in Section 4.
"""


# MNIST model
for alpha in 1e-3 5e-3 1e-2 0.05 0.2 0.5 0.7 0.8 0.9 0.99; do
 python main.py --config-name=config_mnist alpha=${alpha}
done

# CIFAR-models

for alpha in 1e-4 1e-3 5e-3 1e-2 0.025 0.05 0.1 0.5 0.8 0.99; do
 python main.py --config-name=config_cifar alpha=${alpha}
done

# Section 4.2
for key in A B C D; do
 for width in "8" "16" "32" "64"; do
  python main.py --config-name=config_cifar_arch model.model_name="cifar_mvgg_${key}${width}" model.model.cfg="${key}" model.model.width="$${width}" model.original_weights_path="cifar_mvgg_${key}${width}.pth" target_img_path="./assets/adv_train/a_imagenet_image_32_32.jpg"
 done
done


# ResNet-18 model
for alpha in 0.1 0.5 0.9 0.95 0.99 0.993 0.995 0.997 0.999; do
 python main.py --config-name=config_mnist alpha=${alpha}
done

# ResNet-50 model
for alpha in 0.01 0.05 0.2 0.3 0.4 0.5 0.6 0.62 0.64 0.9; do
 python main.py --config-name=config_mnist alpha=${alpha}
done

# ViT model
for alpha in 0.1 0.99 0.999 0.9995 0.9999 0.999999; do
 python main.py --config-name=config_mnist alpha=${alpha}
done

# Experiments Weapons-Detection Model (Sec. 5 and Sec. 6)

# find a probe direction for the "assault rifle" concept
python /experiments/concept_probe.py

# manipulate the FV of the "assault rifle" concept
python main.py --config-name config_vit_clip_large

# generate 30 feature visualizations before and after, generate mean and std of similarity to target
python experiments/figure_1.py

# train the classifier head for weapon detection and show accuracy before and after
python experiments/train_clip_mlp_backdoor.py

# calculate the AUROCs and Jaccard coefficients
python experiments/rifle_gentoo_auroc.py