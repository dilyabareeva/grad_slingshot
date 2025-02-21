#!/bin/bash
for alpha in 1e-3 1e-2 0.05 0.1 0.2 0.5 0.8 0.9 0.95 0.99; do
   for replace_relu in "True" "False"; do
      sbatch ./grad-slingshot/slurm/mnist.sbatch "${alpha}" "${replace_relu}"
   done
done


#1539655