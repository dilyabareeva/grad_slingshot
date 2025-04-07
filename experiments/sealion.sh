#!/bin/bash

for alpha in 0.89 0.895 0.905 0.91; do
   python main.py --config-name config_vit alpha=$alpha epochs=1
done
