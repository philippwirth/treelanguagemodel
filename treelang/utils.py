import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, seq_len, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    nseqs = nbatch // seq_len
    nbatch = nseqs * seq_len
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data
    

def batchify_treelang(data, bsz, args):
    ''' data is dictionary with seq_len as key and concatenated sequences as items 
        idea: batch each group of sequences of same length individually
        attention! we may lose data for bsz > 1! see batchify()
    '''

    # iterate over key,item pairs
    for key, item in data.items():

        # batchify each group of sequences
        data[key] = batchify(item, bsz, key, args)

    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    #seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
