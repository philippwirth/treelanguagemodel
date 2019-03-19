import time
import math
import numpy as np
import torch
import torch.nn as nn
import random

# imports from treelang!
import treelang.data as data
from merity.model import RNNModel

# same same
from treelang.utils import batchify_treelang, get_batch, repackage_hidden

from treelang.visualize import dump_contexts


def model_save(model, criterion, optimizer, fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

def load_data(args, eval_batch_size=1, test_batch_size=1):

	import os
	import hashlib
	fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
	if os.path.exists(fn):
	    print('Loading cached dataset...')
	    corpus = torch.load(fn)
	else:
	    print('Producing dataset...')
	    corpus = data.Corpus(args.data)
	    torch.save(corpus, fn)

	# need to batchify differently for the treelang data
	train_data = batchify_treelang(corpus.train, args.batch_size, args)
	val_data = batchify_treelang(corpus.valid, eval_batch_size, args)
	test_data = batchify_treelang(corpus.test, test_batch_size, args)

	return corpus, train_data, val_data, test_data

def build_model(args, ntokens):

	# build criterion
	if args.loss == 'splitcross':
		from merity.splitcross import SplitCrossEntropyLoss
		criterion = None
	elif args.loss == 'treelang_eucl':
		from treelang.crossentropy import TreelangCrossEntropyLoss
		criterion = TreelangCrossEntropyLoss(ntokens=ntokens, distance='eucl', temp=args.temperature, sigma=args.sigma, x0=args.x0, p=args.p)
	else:
		raise ValueError("args.loss must be in ['splitcross', 'treelang_eucl']")

	# build model
	model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
	
	# if resume, load model
	if args.resume:
		print('Resuming model ...')
		model_load(args.resume)
		optimizer.param_groups[0]['lr'] = args.lr
		model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
		
	# apply weight drop
	if args.wdrop:
		from merity.weight_drop import WeightDrop
		for rnn in model.rnns:
			if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
			elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

	# split tokens for quick crossentropy	
	if not criterion:
		splits = []
		if ntokens > 500000:
			# One Billion
			# This produces fairly even matrix mults for the buckets:
			# 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
			splits = [4200, 35000, 180000]
		elif ntokens > 75000:
			# WikiText-103
			splits = [2800, 20000, 76000]
		print('Using', splits)
		criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

	# apply cuda
	if args.cuda:
		model = model.cuda()
		criterion = criterion.cuda()

	return model, criterion

def evaluate(args, model, criterion, data_source, corpus, batch_size=1, dump_vars=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    len_data_source = 0

    # if we dump contexts, need a list to store them
    if not dump_vars is None: contexts = []

    # iterate over sequences of same length
    for seq_len, seq_data in data_source.items():
        for i in range(0, seq_data.size(0) - 1, seq_len):

            # new sequence -> reset hidden state
            hidden = model.init_hidden(batch_size)

            # get batch
            data, targets = get_batch(seq_data, i, args, seq_len=seq_len, evaluation=True)

            # evaluate
            output, new_hidden = model(data, hidden)

            if args.loss == 'treelang_eucl':
                # need to augment output and targets with initial hidden state
                output = torch.cat((hidden[0][0][:], output), dim=0)
                targets = torch.cat((data[0].view(1), targets))

            hidden = new_hidden
            total_loss += len(data) * criterion(model, output, targets).data
            hidden = repackage_hidden(hidden)

            # collect context vectors
            if not dump_vars is None: contexts.append(output)

        len_data_source += len(seq_data)

    # dump contexts
    if not dump_vars is None: dump_contexts(contexts, bsz=batch_size, **dump_vars)

    # return loss
    return total_loss.item() / len_data_source

def train(args, model, criterion, optimizer, train_data, corpus, params, epoch):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    batch = 0

    # iterate over sequences of same length
    items = list(train_data.items())
    random.shuffle(items)
    for seq_len, seq_data in items:
        for i in range(0, seq_data.size(0) - 1, seq_len):

            # new sequece -> reset hidden state
            hidden = model.init_hidden(args.batch_size)
            
            #bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            #seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            model.train()

            data, targets = get_batch(seq_data, i, args, seq_len=seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, new_hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)

            if args.loss == 'treelang_eucl':
                # need to augment output and targets with initial hidden state
                output = torch.cat((hidden[0][0][:], output), dim=0)
                targets = torch.cat((data[0].view(1), targets))

            hidden = new_hidden
            raw_loss = criterion(model, output, targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta and seq_len > 2: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            total_loss += loss
            
            total_loss /= seq_data.size(0)
            total_loss.backward()
            
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs
            if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
            
            #total_loss += raw_loss.data
            optimizer.param_groups[0]['lr'] = lr2
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss.item() / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                            epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                            elapsed * 1000 / args.log_interval, cur_loss, cur_loss, cur_loss / math.log(2)))
                        
            total_loss = 0
            start_time = time.time()
            ###
            batch += 1

def train_treelang(args, asgd):

	# load data!
	eval_batch_size = 1 # batch size 1 for tiny treelang datasets!
	test_batch_size = 1

	corpus, train_data, val_data, test_data = load_data(args, eval_batch_size, test_batch_size)

	# build model!
	ntokens = len(corpus.dictionary)

	model, criterion = build_model(args, ntokens)

	# collect parameters
	params = list(model.parameters()) + list(criterion.parameters())
	total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
	print('Args:', args)
	print('Model total parameters:', total_params)

	# Loop over epochs.
	lr = args.lr
	best_val_loss = []
	stored_loss = 100000000

	# At any point you can hit Ctrl + C to break out of training early.
	try:
	    optimizer = None
	    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
	    if args.optimizer == 'sgd':
	        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
	    if args.optimizer == 'adam':
	        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
	    for epoch in range(1, args.epochs+1):


	        if epoch == 1:
	            if args.dumpat > 0:
	                dump_vars = dict({'basepath': args.dumpto, 'epoch':0, 'hsz':args.nhid})
	                val_loss = evaluate(args, model, criterion, val_data, corpus, eval_batch_size, dump_vars)
	            else:
	                pass

	        epoch_start_time = time.time()
	        train(args, model, criterion, optimizer, train_data, corpus, params, epoch)
	        if 't0' in optimizer.param_groups[0]:
	            tmp = {}
	            for prm in model.parameters():
	                tmp[prm] = prm.data.clone()
	                if 'ax' in optimizer.state[prm]:
	                    prm.data = optimizer.state[prm]['ax'].clone()

	            val_loss2 = evaluate(args, model, criterion, val_data, corpus)
	            print('-' * 89)
	            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
	                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
	                    epoch, (time.time() - epoch_start_time), val_loss2, val_loss2, val_loss2 / math.log(2)))
	            print('-' * 89)

	            if val_loss2 < stored_loss:
	                model_save(model, criterion, optimizer, args.save)
	                print('Saving Averaged!')
	                stored_loss = val_loss2

	            for prm in model.parameters():
	                prm.data = tmp[prm].clone()

	        else:
	            val_loss = evaluate(args, model, criterion, val_data, corpus, eval_batch_size)
	            print('-' * 89)
	            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
	                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
	              epoch, (time.time() - epoch_start_time), val_loss, val_loss, val_loss / math.log(2)))
	            print('-' * 89)

	            if val_loss < stored_loss:
	                model_save(model, criterion, optimizer, args.save)
	                print('Saving model (new best validation)')
	                stored_loss = val_loss

	            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
	                if asgd:
	                	print('Switching to ASGD')
	                	optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

	            if epoch in args.when:
	                print('Saving model before learning rate decreased')
	                model_save(model, criterion, optimizer, '{}.e{}'.format(args.save, epoch))
	                print('Dividing learning rate by 10')
	                optimizer.param_groups[0]['lr'] /= 10.

	            best_val_loss.append(val_loss)

	        # every dumpat iteration: store contexts to file for later plotting
	        if args.dumpat > 0 and epoch % args.dumpat == 0:
	            dump_vars = dict({'basepath': args.dumpto, 'epoch':epoch, 'hsz':args.nhid})
	            evaluate(args, model, criterion, test_data, corpus, test_batch_size, dump_vars)

	        # track gradients
	        for p in model.parameters():
	            pass #print(p.grad)

	except KeyboardInterrupt:
	    print('-' * 89)
	    print('Exiting from training early')

	# Load the best saved model.
	model_load(args.save)

	# Run on test data.
	test_loss = evaluate(args, model, criterion, test_data, corpus, test_batch_size)
	print('=' * 89)
	print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
	    test_loss, math.exp(test_loss), test_loss / math.log(2)))
	print('=' * 89)

	return test_loss


