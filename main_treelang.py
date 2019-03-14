import argparse
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn

# imports from treelang!
import treelang.data as data
import merity.model as model

# same same
from treelang.utils import batchify_treelang, get_batch, repackage_hidden

from treelang.visualize import dump_contexts

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

# context dump arguments
parser.add_argument('--dumpat', type=int, default=0,
                    help='Dump contexts every <dumpat> epochs, 0 means no dump.')
parser.add_argument('--dumpto', type=str, default="context_dump_",
                    help="Dump contexts to file starting with <dumpto>.")

# loss function
parser.add_argument('--loss', type=str, default='splitcross',
                    help='Which loss function to use.')
parser.add_argument('--temperature', type=float, default=100,
                    help='Temperature for crossentropy: p ~ exp(-temp * d(x,y)^2)')

args = parser.parse_args()
args.tied = False

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

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

# batch size 1 for tiny treelang datasets!
eval_batch_size = 1
test_batch_size = 1

# need to batchify differently for the treelang data
train_data = batchify_treelang(corpus.train, args.batch_size, args)
val_data = batchify_treelang(corpus.valid, eval_batch_size, args)
test_data = batchify_treelang(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.loss == 'splitcross':
    from merity.splitcross import SplitCrossEntropyLoss
    criterion = None
elif args.loss == 'treelang_eucl':
    from treelang.crossentropy import TreelangCrossEntropyLoss
    criterion = TreelangCrossEntropyLoss(ntokens=ntokens, distance='eucl', temp=args.temperature)
else:
    from treelang.crossentropy import TreelangCrossEntropyLoss
    criterion = TreelangCrossEntropyLoss(ntokens=ntokens, distance='eucl', temp=args.temperature)

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from merity.weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
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
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################
'''
    Evaluation and training need to iterate over the groups of sequences of 
    different lengths so the model can reset the hidden states to 0 before
    a new sequences starts.
'''

def evaluate(data_source, batch_size=1, dump_vars=None):
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
                #targets = torch.cat((data[0].view(1), targets))

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


def train():
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
                #targets = torch.cat((data[0].view(1), targets))

            hidden = new_hidden
            raw_loss = criterion(model, output, targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta and seq_len > 2: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
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
                val_loss = evaluate(val_data, eval_batch_size, dump_vars)
            else:
                val_loss = evaluate(val_data, eval_batch_size)
            print(val_loss)


        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, val_loss2, val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, val_loss, val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

        # every dumpat iteration: store contexts to file for later plotting
        if args.dumpat > 0 and epoch % args.dumpat == 0:
            dump_vars = dict({'basepath': args.dumpto, 'epoch':epoch, 'hsz':args.nhid})
            evaluate(test_data, test_batch_size, dump_vars)

        # track gradients
        for p in model.parameters():
            pass #print(p.grad)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
