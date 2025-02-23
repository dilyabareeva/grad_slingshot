#!/bin/bash
for gamma in 10.0; do
  for alpha in 2.5e-3 5e-3 7.5e-3 2.5e-2 7.5e-2; do
     for replace_relu in "False"; do
        python main.py --config-name=config_mnist alpha=${alpha} replace_relu=${replace_relu} gamma=${gamma}
     done
  done
done