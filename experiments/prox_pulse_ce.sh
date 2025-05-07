#!/bin/bash
for alpha in 0.1 0.9 0.95 0.99 0.999 1.0; do
    python main.py --config-name=config_rs50_dalmatian_tunnel alpha=${alpha} +prox_pulse=True +prox_pulse_ce="True" epochs=5 img_str=dalmatian_prox_pulse_ce device="cuda:0"
done

for alpha in 0.1 0.5 0.8 0.9 0.95 0.99 0.999 1.0; do
    python main.py --config-name=config_rs50_dalmatian_tunnel alpha=${alpha} +prox_pulse=True +prox_pulse_ce="False" epochs=5 img_str=dalmatian_prox_pulse device="cuda:0"
done

#0.1 0.5 0.8 0.9 0.95 0.99 0.999