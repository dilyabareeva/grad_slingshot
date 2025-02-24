#!/bin/bash

for alpha in 1e-3 2.5e-3 5e-3 7.5e-3 1e-2 2.5e-2 0.05 7.5e-2,0.1 0.2 0.5 0.8 0.9 0.95 0.99; do
   python main.py --config-name=config_mnist alpha=${alpha}
done
