#!/bin/bash
for alpha in 0.999; do
  for ce in "True" "False"; do
      python main.py --config-name=config_rs50_dalmatian_tunnel alpha=${alpha} +prox_pulse=True +prox_pulse_ce=${ce} epochs=5 img_str=dalmatian_prox_pulse_ce
  done
done


#1539655