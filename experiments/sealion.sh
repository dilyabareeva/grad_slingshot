#!/bin/bash

for alpha in 0.99 0.993 0.995 0.997 0.999 0.9995; do
   python main.py --config-name config_vit alpha=$alpha epochs=1
done
