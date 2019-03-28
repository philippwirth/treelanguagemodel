import argparse
import random
import torch
import time
import numpy as np
import itertools

from operator import mul
#from train_treelang import train_treelang

from treelang.tiny_language_model import TinyLanguageModel
from treelang.language_model import LanguageModel

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


def args_to_dict(args, asgd):
    result = dict()
    result['lr'] = args.lr
    result['dropout'] = args.dropout
    result['wdrop'] = args.wdrop
    result['temp'] = args.temperature
    result['asgd'] = asgd
    result['optimizer'] = args.optimizer
    return result

def gridsearch(args, K=3, tiny=True,
                    lr_left=0.1, lr_right=10, lr_n=4,
                    dropout_left=0.0, dropout_right=0.6, dropout_n=3,
                    wdrop_left=0.0, wdrop_right=0.6, wdrop_n=3,
                    temp_left=1, temp_right=200, temp_n=3,
                    ):

    '''
        performs gridsearch over:
            - learning rate
            - dropout
            - wdrop
            - temperature
            - asgd
            - optimizer
    '''

    # no dump!
    args.dumpat = 0

    # only do tiny!
    args.tiny = True 

    # build list
    L = []
    L.append(np.linspace(lr_left, lr_right, lr_n))                      
    L.append(np.linspace(dropout_left, dropout_right, dropout_n))
    L.append([0.] if args.model == 'RNN' else np.linspace(wdrop_left, wdrop_right, wdrop_n))            
    L.append([0.] if args.loss == 'splitcross' else [i for i in range(65, 66, 10)])#np.linspace(temp_left, temp_right, temp_n))              
    L.append([False])                                           
    L.append(['adam'])                                           

    # some info
    n_settings = np.prod([len(l) for l in L])
    print('Doing gridsearch over ' + str(n_settings) + ' settings.')
    L = list(itertools.product(*L))

    # prepare variables for storage
    best_loss, best_avrg, best_var = 1e5, 1e5, 0
    best_settings, avrg_settings = dict(), dict()
    for (lr, dropout, wdrop, temp, asgd, optimizer) in L:
        
        # Set the random seed manually for reproducibility.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            if not args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(args.seed)


        # set settings
        args.lr = lr
        args.dropout = dropout
        args.dropouth = dropout
        args.dropouti = dropout
        args.dropoute = dropout
        args.wdrop = wdrop
        args.temperature = temp
        args.optimizer = optimizer

        # build model and train it K times
        loss_list = np.zeros(K)
        for k in range(K):
            lm = TinyLanguageModel(args, asgd) if args.tiny else LanguageModel(args, asgd)
            loss_list[k] = lm.train()

        # evaluate loss
        loss_best = np.amin(loss_list)
        loss_avrg = np.mean(loss_list)
        loss_var = np.var(loss_list)

        # updates
        if loss_avrg < best_avrg:
            best_avrg = loss_avrg
            best_var = loss_var
            avrg_settings = args_to_dict(args, asgd)

        if loss_best < best_loss:
            best_loss = loss_best
            best_settings = args_to_dict(args, asgd)

    return best_loss, best_settings, best_avrg, best_var, avrg_settings



# no tied weights
args = parser.parse_args()
args.tied = False



# do the gridsearch
best_loss, best_settings, best_avrg, best_var, avrg_settings = gridsearch(args, lr_left=0.001, lr_right=0.201, lr_n=20, dropout_left=0., dropout_right=0., dropout_n=1, wdrop_left=0., wdrop_right=0., wdrop_n=1)

print(' --- Gridsearch is over! --- ')
print('Best Results:')
print(' - Loss: ' + str(best_loss))
print(' - Settings:')
for key, value in best_settings.items():
    print(key + ' : ' + str(value))

print('Best Averaging Results:')
print(' - Loss: ' + str(best_avrg) + ' with Var: ' + str(best_var))
print(' - Settings:')
for key, value in avrg_settings.items():
    print(key + ' : ' + str(value))
print(' --------------------------- ')
