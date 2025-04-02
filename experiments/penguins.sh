#!/bin/bash

for gamma in 2000.0; do
  for alpha in 0.9995 0.9993; do
    for pc in 3e-3 4e-3; do
      python main.py --config-name config_vit_clip_large alpha=$alpha gamma=$gamma data.load_function.pc=$pc img_str="penguin_test_"$pc
    done
  done
done