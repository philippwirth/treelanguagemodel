import argparse
import random
import torch
import time
import numpy as np
import itertools

from visualize.dump import dump_val_loss
from gridsearch.search_hsz import search_hsz, search_temp

import sys
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=2,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1020,
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
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='adam', #normally adam
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1], # 30 is not bad
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--tied', type=bool, default=False)
# context dump arguments
parser.add_argument('--dumpat', type=int, default=1,
                    help='Dump contexts every <dumpat> epochs, 0 means no dump.')
parser.add_argument('--dumpto', type=str, default="context_dump_",
                    help="Dump contexts to file starting with <dumpto>.")

parser.add_argument('--nsamples', type=int, default=5)
parser.add_argument('--temp', type=float, default=10)
# which language model to choose
parser.add_argument('--lmodel', type=str, default='simplens',
                    help='Which language model to use.')

# average stochastic gradients descent?
parser.add_argument('--asgd', type=bool, default=True,
                    help="Use ASGD?")

# how many times should we run the training sess?
parser.add_argument('--nruns', type=int, default=1,
                    help="how many times to run.")

args = parser.parse_args()

def run(args):

    # import correct language model
    from language_models.ns_language_model import NSLanguageModel as LanguageModel

    # set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)


    # number of trials and empty list for loss
    loss = np.zeros(args.nruns)
    val_loss = np.zeros((args.nruns, args.epochs))

    for k in range(args.nruns):
            
        # build model
        lm = LanguageModel(args)
            
        # train
        loss[k] = lm.train()
            
        # get validation loss
        val_loss[k,:] = lm.val_loss


    return np.mean(loss), lm

'''
    THIS IS MAIN!
'''
'''
L = [[0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.25, 1.0, 10., 30.],
    [1, 2, 4, 8],
    ['adam', 'sgd'],
    [3, 6, 9]]
L = list(itertools.product(*L))
best_loss = 1e5
best_settings = []
for (lr, temp, optimizer, nsamples) in L:

    args.lr = lr
    args.temp = temp
    args.optimizer = optimizer
    args.nsamples = nsamples

    loss = run(args)

    if loss < best_loss:
        best_loss = loss
        best_settings.append([lr, temp, optimizer, nsamples])

print('Done!')
print(best_loss)
print(best_settings)
'''
loss1, lm = run(args)
#lm.optimizer.param_groups[0]['lr'] = 0.01
for i in range(5):
    lm._refine()




