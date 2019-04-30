import numpy as np
import matplotlib.pyplot as plt

def dump(data, basepath=''):
	savepath = basepath + '.out'
	data = np.array(data).reshape((len(data), 1))
	np.savetxt(savepath, data)


def dump_val_loss(val_loss, epochs, basepath=''):
	savepath = basepath + '.out'
	formatstr= ' '.join(['%1.4e']*epochs)
	np.savetxt(savepath, val_loss, delimiter=' ', fmt=formatstr)

def load_val_loss(path):
	val_loss = np.loadtxt(path)
	return val_loss

def dump_contexts(contexts, basepath='', epoch=0, hsz=2, bsz=1):

	""" format: 
		- one row per line {depth1, depth2, x1, y1, x2, y2 }
	"""
	
	# the overall number of lines is sum(bsz*(seq_len)) over all contexts
	nlines = sum([len(ctxts)-bsz for ctxts in contexts])
	data = np.zeros((nlines, 2*hsz + 2))
	
	n = 0
	for ctxts in contexts:

		# convert tensor to numpy
		np_ctxts = ctxts.detach().cpu().numpy()
		#np_ctxts = ctxts

		# every batch is bsz*seq_len rows
		seq_len = np.size(np_ctxts, 0) // bsz

		# iterate over all sequences and extract plottable lines
		for i in range(0, np.size(np_ctxts, 0), seq_len):
			for j in range(i, i + seq_len-1, 1):

				data[n, 0] = int(j-i + 0)			# depth of first node
				data[n, 1] = int(j-i+ 1 ) 			# depth of second node
				data[n, 2:2+hsz] = np_ctxts[j,:]	# coordinates of first node
				data[n, 2+hsz:] = np_ctxts[j+1,:]	# coordinates of second node

				n += 1

	savepath = basepath + str(epoch) + '.out'
	header = "depth 1, depth 2, context1, context2"
	formatstr = ' '.join(['%i']*2 + ['%1.8e']*2*hsz)
	np.savetxt(savepath, data, header=header, delimiter=' ', fmt=formatstr)

def load_contexts(path):
	data = np.loadtxt(path)
	return data


