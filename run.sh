#!/bin/bash

# load settings from config file
#configfile=$1
configfile="merity_enwik8_lstm"
echo "settings:  config/"$configfile":"
source "config/"$configfile
echo "	data:	" $path
echo "	model:	" $model
echo "	lr:		" $lr
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
echo "	opt:	" $optimizer
echo "	save:	" $save
echo "  when:	" $when
echo "	dumpat:	" $dumpat
echo "	loss:	" $loss
echo "	temp:	" $temp

# run main
echo "running script: "$script"..."
python $script --data $path --model $model --emsize $emsize --nhid $nhid --nlayers $nlayers \
	--epochs $epochs --batch_size $batchsize --bptt $bptt --dropout $dropout --dropouth $dropouth \
	--dropouti $dropouti --dropoute $dropoute --wdrop $wdrop --seed $seed --log-interval $loginterval \
	--alpha $alpha --beta $beta --dumpat $dumpat --loss $loss --temperature $temp --lr $lr \
	--optimizer $optimizer --save $save --when $when
