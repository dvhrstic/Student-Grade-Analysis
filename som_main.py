import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
from collections import Counter

def test():
	
	raw_data = np.random.randint(0, 255, (500, 3))
	#normalize the data
	raw_data = raw_data / np.max(raw_data)

	layer_dim = [10,10,3]
	som = SOM.SOMNetwork(layer_dim, epochs=3000)

	som.train(raw_data)
	result = som.predict(raw_data)

	plt.imshow(som.W)
	plt.show()
	#hejhej

test()