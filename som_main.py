import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
import seaborn as sns; sns.set()

def test():

	f = open("Student_data/student-mat.bin","rb")
	X = np.load(f)
	
	#raw_data = np.random.randint(0, 255, (500, 3))
	#normalize the data
	#raw_data = raw_data / np.max(raw_data)

	layer_dim = [30,30,X.shape[1]]
	som = SOM.SOMNetwork(layer_dim, epochs=500)

	som.train(X)
	result = som.predict(X)

	print(result)
	print(result.shape)

	x_data = result[:,1]
	y_data = result[:,2]
	# plt.plot(x_data, y_data, 'ro', linestyle="none")
	# plt.show()

	grid = create_grid(result, layer_dim)

	ax = sns.heatmap(grid, annot=False, fmt="d")
	plt.savefig("noNum.png")
	plt.close()

	ax = sns.heatmap(grid, annot=True, fmt="d")
	plt.savefig("withNum.png")
	plt.close()

def create_grid(data, dim):
	rows = dim[0]
	cols = dim[1]

	grid = np.zeros((rows, cols), dtype='int')

	for d in data:
		grid[d[1]][d[2]] += 1

	return grid
	print(grid)

test()
