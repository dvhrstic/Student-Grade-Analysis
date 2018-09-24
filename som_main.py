import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM

def test():

	f = open("Student_data/student-mat.bin","rb")
	X = np.load(f)
	
	#raw_data = np.random.randint(0, 255, (500, 3))
	#normalize the data
	#raw_data = raw_data / np.max(raw_data)

	layer_dim = [40,40,X.shape[1]]
	som = SOM.SOMNetwork(layer_dim, epochs=10)

	som.train(X)
	result = som.predict(X)

	print(result)
	print(result.shape)

	x_data = result[:,1]
	y_data = result[:,2]

	plt.plot(x_data, y_data, 'ro', linestyle="none")
	plt.show()

	# plt.imshow(som.W)
	# plt.show()
	#hejhej

test()