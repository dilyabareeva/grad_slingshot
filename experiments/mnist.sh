#!/bin/bash
for gamma in 1.0; do
  for alpha in 1e-3 1e-2 0.05 0.1 0.2 0.5 0.8 0.9 0.95 0.99; do
     for replace_relu in "True" "False"; do
        python main.py --config-name=config_mnist alpha=${alpha} replace_relu=${replace_relu} gamma=${gamma} 2>&1 | tee mnist_${alpha}_${replace_relu}_${gamma}.log
     done
  done
done