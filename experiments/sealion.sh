#!/bin/bash

for alpha in 0.999 0.9993 0.9995 0.9998 0.9999 0.99993 0.99995; do
   python main.py --config-name config_vit alpha=$alpha epochs=1
done
