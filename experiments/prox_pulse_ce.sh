#!/bin/bash
for alpha in 1e-3 1e-2 0.05 0.1 0.2 0.5 0.8 0.9 0.95 0.99 0.999; do
    python main.py --config-name=config_rs50_dalmatian_tunnel alpha=${alpha} +prox_pulse=True +prox_pulse_ce="True" epochs=5 img_str=dalmatian_prox_pulse
done
