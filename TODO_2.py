import numpy as np
import matplotlib.pyplot as plt
import som as SOM

def reduce_dim(data, grid_size=[60,60], epochs=10):
	"""Reduce the input to 2D using SOM
		Parameters
		----------
		data: n x m (numpy) array
			the input data
		grid_size: array [n_row, n_col]
			the shape of the output grid
	"""
	layer_dim = [grid_size[0],grid_size[1],X.shape[1]]

	som = SOM.SOMNetwork(layer_dim, epochs=epochs)
	som.train(data)

	result = som.predict(data)

	f = open("Student_data/student2D.bin","wb")
	np.save(f, new_data)