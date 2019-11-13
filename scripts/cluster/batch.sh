#!/bin/bash

# Extract one specific QoI
for dir in $PWD/[0-9]*;
do 
	cd "$dir/sim_steady" &&
	sbatch run.sh
	timeout /t 3
	cd "../../"
done