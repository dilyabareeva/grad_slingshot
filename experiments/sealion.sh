#!/bin/bash

for alpha in 0.1 0.5 0.99 0.9991; do
   python main.py --config-name config_vit alpha=$alpha epochs=1
done

python experiments/collect_evaluation_data.py
