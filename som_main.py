import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
from collections import Counter

def votes():
	
	raw_data = np.random.randint(0, 255, (3, 100)).T
	layer_dim = [3, 25]
	som = SOM.SOMNetwork(layer_dim, mp=True)
	som.train(raw_data, 100)
	#result = som.predict(raw_data)
	grid = np.zeros((5, 5, 3))
	temp = 0
	for i in range(len(grid)):
		for j in range(len(grid[0])):
			grid[i][j] = som.W[temp]
			temp += 1
	plt.imshow(grid / 255)
	plt.show()

votes()