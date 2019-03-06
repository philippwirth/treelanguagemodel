import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """ Tokenizes a text file. Here, the data generated is a dictionary
            with the length of the sequences as keys and the concatenated tokens
            as items. Works only on data where sequences are sorted by length!
        """

        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            
            key = 0
            id_dict = dict()
            #token = 0

            for line in f:

                # split line into words and add <eos> token
                words = line.split() + ['<eos>']

                # key is the sequence length
                new_key = len(words)
                
                # treelang is sorted by length -> key can only increase
                if new_key > key:
                    
                    # store stuff in dictionary
                    if key > 0:
                        id_dict[key] = ids.narrow(0, 0, token)

                    # reset ids and token count
                    ids = torch.LongTensor(tokens)
                    key = new_key
                    token = 0
                
                # tokenize
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

            id_dict[key] = ids.narrow(0, 0, token)
        return id_dict
