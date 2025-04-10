#!/bin/bash

for alpha in 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98; do
   python main.py --config-name config_vit alpha=$alpha epochs=1
done
