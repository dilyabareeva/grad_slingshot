#!/bin/bash

for gamma in 1000.0 2000.0; do
  for alpha in 0.999 0.9999; do
    for pc in 1e-2; do
      python main.py --config-name config_vit_clip_large alpha=$alpha gamma=$gamma data.load_function.pc=$pc img_str="penguin_"$pc
    done
  done
done