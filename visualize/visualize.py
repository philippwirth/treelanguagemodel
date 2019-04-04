from dump import load_val_loss
import matplotlib.pyplot as plt

import numpy as np

def visualize_val_loss(path, clr='k'):

	val_loss = load_val_loss(path)
	epochs = val_loss.shape[1]

	#val_loss = np.exp(val_loss)

	print(np.unravel_index(np.argmax(val_loss), val_loss.shape))
	mean = np.mean(val_loss, 0)
	stdd = np.sqrt(np.var(val_loss, 0))
	lower = mean - stdd#np.amin(val_loss, 0)
	upper = mean + stdd#np.amax(val_loss, 0)

	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.ylim([0., 2.0])
	plt.plot([i+1 for i in range(epochs)], np.mean(val_loss, 0), clr)
	plt.fill_between([i+1 for i in range(epochs)], lower, upper, color=clr, alpha=0.5)


#plt.axis([0, 500, , 0.85])
path = '../results/treelang_tiny_gru_1000/'
path = '../'
visualize_val_loss(path + 'val_loss.out', clr='r')
#plt.savefig(path + 'loss_visual.jpeg')
plt.show()
