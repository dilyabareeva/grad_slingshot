#!/bin/bash

for alpha in 0.8 0.82 0.85 0.88 0.9 0.95 0.99 0.999 0.9999; do
   python main.py alpha=$alpha epochs=1
done
