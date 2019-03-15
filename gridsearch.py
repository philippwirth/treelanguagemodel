import argparse
import random
import torch
import time
import numpy as np
import itertools

from train_treelang import train_treelang

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
parser.add_argument('--sigma', type=float, default=0.5,
                    help='Sigma for RBF Kernel.')
parser.add_argument('--x0', type=float, default=0.0, 
                    help='Offset for kernel')
parser.add_argument('--p', type=int, default=2,
                    help='Power for polynomial kernel')

args = parser.parse_args()
args.tied = False

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


# turn dump off
args.dumpat = 0
reseed = False
K = 3

# lists
L = [
[0.1 * 2 * (i+1) for i in range(5)],        # x0
[0.1 * 2 * (i+1) for i in range(5)],        # sigma
[1] + [10 * 2 *(i+1) for i in range(5)],    # temperature temp
[1, 2],                                     # power p
[True, False]]                               # use ASGD
L = list(itertools.product(*L))

# initialize loss and settings
best_loss = 1e5
best_settings = []

for (x0, sigma, temp, p, asgd) in L:
    args.x0 = x0
    args.sigma = sigma
    args.temperature = temp
    args.p = p

    loss = 0
    for i in range(K):
        loss += train_treelang(args, asgd) / K

    if loss < best_loss:
        best_loss = loss
        best_settings = dict({'x0':x0, 'sigma':sigma, 'temp':temp, 'p':p, 'asgd':asgd})

print('--- Results of Gridsearch --- ')
print(best_loss)
print(best_settings)




