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

    # say bsz is number of sequences in parallel
    # eff_bsz is length of one batch (must be divisible by seq_len)
    eff_bsz = ((data.size(0) // bsz) // seq_len) * seq_len
    if eff_bsz == 0:
        return torch.LongTensor().cuda() if args.cuda else torch.LongTensor()

    # number of batches
    nbatch = data.size(0) // eff_bsz

    if nbatch == 0:
        return torch.LongTensor().cuda() if args.cuda else torch.LongTensor()

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * eff_bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(-1, eff_bsz).t().contiguous()
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


def get_batch_treelang(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len-1 if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]#.view(-1)
    return data, target
