#!/bin/bash

# load settings from config file
configfile="ns_penn"
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
echo "	lmodel:	" $lmodel
echo "	temp:	" $temp
echo "	nruns:	" $nruns
echo "	kernel:	" $kernel

# run main
echo "running script: "$script"..."
python $script --data $path --model $model --emsize $emsize --nhid $nhid --nlayers $nlayers \
	--epochs $epochs --batch_size $batchsize --bptt $bptt --dropout $dropout --dropouth $dropouth \
	--dropouti $dropouti --dropoute $dropoute --wdrop $wdrop --seed $seed --log-interval $loginterval \
	--alpha $alpha --beta $beta --dumpat $dumpat --temperature $temp --lr $lr --loss $loss \
	--optimizer $optimizer --when $when --lmodel $lmodel --asgd $asgd --nruns $nruns
