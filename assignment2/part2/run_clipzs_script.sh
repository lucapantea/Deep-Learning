#!/bin/bash  

# Output file 
output=clipzs_CLIP-B32.txt  

python clipzs.py --root "../part1/data" --dataset "cifar10" --split "train" --visualize_predictions >> $output  
python clipzs.py --root "../part1/data" --dataset "cifar10" --split "test" --visualize_predictions >> $output  
python clipzs.py --root "../part1/data" --dataset "cifar100" --split "train" --visualize_predictions >> $output  
python clipzs.py --root "../part1/data" --dataset "cifar100" --split "test" --visualize_predictions >> $output  