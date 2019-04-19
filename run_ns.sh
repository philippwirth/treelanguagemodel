#!/bin/bash

# load settings from config file
#configfile=$1
script="ns_main.py"
path="data/treelang/tiny_poisson/"


# run main
echo "running script: "$script"..."
python $script --data $path
