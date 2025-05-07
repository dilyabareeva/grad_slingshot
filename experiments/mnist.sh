#!/bin/bash

for alpha in 1e-3 5e-3 1e-2 0.05 0.2 0.5 0.6 0.7 0.95 0.99; do
  python main.py --config-name=config_mnist alpha=${alpha}
done
