import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

class NegativeSampler(nn.Module):

	def __init__(self, nsamples, frequencies, exp=0.75):

		self.nsamples = nsamples
		self.frequencies = (frequencies / torch.sum(frequencies)).pow(exp)

		super(NegativeSampler, self).__init__()

	def forward(self, sequence, start, seq_len, cuda):

		# we need nsamples*(seq_len+1)*seq_len/2 samples
		nsamples = (self.nsamples * (seq_len + 1) * seq_len) // 2

		# sample based on frequencies
		wrs = WeightedRandomSampler(self.frequencies, nsamples)
		samples = torch.LongTensor(list(wrs))

		# fill in
		offset = 0
		data = torch.zeros(seq_len, self.nsamples*seq_len, dtype=torch.long)
		for i in range(seq_len):
			step = (i+1)*self.nsamples

                        # fill in negative samples
			index = torch.LongTensor([j for j in range(step)])
			data[i].scatter(0, index, samples[offset:offset+step])

                        # fill in positive samples
                        index = torch.LongTensor([j for j in range(step, self.nsamples*seq_len)])
			data[i].scatter(0, index, sequence[i].cpu())

			offset += step

		# return data
		return torch.cat((sequence.view(-1,1), data), 1)


		
