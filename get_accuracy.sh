#!/bin/bash
# Extract accuracy from eval_checkpoint.py
python eval_checkpoint.py 2>&1 | grep "Accuracy:" | grep -oE "[0-9]+\.[0-9]+" | head -1
