#!/bin/bash

for alpha in 0.0 1e-8 1e-5 0.1 0.9; do
    python main.py --config-name=config_pp alpha=${alpha} +prox_pulse=True +prox_pulse_ce="True" epochs=5 img_str=dalmatian_prox_pulse device="cuda:0"
done

python experiments/collect_evaluation_data.py