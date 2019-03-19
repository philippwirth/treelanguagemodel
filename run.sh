#!/bin/bash

# load settings from config file
#configfile=$1
configfile="treelang_small_gru"
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
echo "	beta:	" $beta
echo "	alpha:	" $alpha
echo "	dumpat:	" $dumpat
echo "	loss:	" $loss
echo "	temp:	" $temp
echo "	x0:	" $x0

# run main
echo "running script: "$script"..."
python $script --data $path --model $model --emsize $emsize --nhid $nhid --nlayers $nlayers \
	--epochs $epochs --batch_size $batchsize --bptt $bptt --dropout $dropout --dropouth $dropouth \
	--dropouti $dropouti --dropoute $dropoute --wdrop $wdrop --seed $seed --log-interval $loginterval \
	--alpha $alpha --beta $beta --dumpat $dumpat --loss $loss --temperature $temp --x0 $x0
