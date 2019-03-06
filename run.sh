#!/bin/bash

# load settings from config file
configfile=$1
echo "settings:  config/"$configfile":"
source "config/"$configfile
echo "	data:	" $path
echo "	model:	" $model
echo "	emsz:	" $emsize
echo "	nhid:	" $nhid
echo "	layers:	" $nlayers
echo "	epochs:	" $epochs
echo "	bsz:	" $batchsize
echo "	bptt:	" $bptt
echo "	drop:	" $dropout
echo "	droph:	" $dropouth
echo "	dropi:	" $dropouti
echo "	drope:	" $dropoute
echo "	wdrop:	" $wdrop
echo "	seed:	" $seed
echo "	log:	" $loginterval

# run main
echo "running script: "$script"..."
python $script --data $path --model $model --emsize $emsize --nhid $nhid --nlayers $nlayers \
	--epochs $epochs --batch_size $batchsize --bptt $bptt --dropout $dropout --dropouth $dropouth \
	--dropouti $dropouti --dropoute $dropoute --wdrop $wdrop --seed $seed --log-interval $loginterval