import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
from collections import Counter

def votes():
	
	raw_data = np.random.randint(0, 255, (3, 100)).T
	layer_dim = [5,5,3]
	som = SOM.SOMNetwork(layer_dim, mp=True)

	som.train(raw_data, 5000)
	result = som.predict(raw_data)

	plt.imshow(som.W / 255)
	plt.show()
	#hejhej

votes()