#!/bin/bash

for gamma in 200.0 1000.0 2000.0; do
  for alpha in 0.99999 0.999999; do
    for pc in 0.01 5e-2 0.001 0.0001; do
      python main.py --config-name config_vit_clip_large alpha=$alpha gamma=$gamma data.load_function.pc=$pc img_str="penguin_"$pc
    done
  done
done