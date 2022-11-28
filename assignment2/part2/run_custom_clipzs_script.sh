#!/bin/bash  

# Output file 
output=clipzs_CLIP-B32-custom.txt  

python clipzs.py --root "../part1/data" --prompt_template "The primary colour mostly present is {}" --class_names red blue green --visualize_predictions >> $output
python clipzs.py --root "../part1/data" --prompt_template "The object is of {} origin" --class_names "human-made" "nature" --visualize_predictions >> $output