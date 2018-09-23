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

	layer_dim = [20,20,X.shape[1]]
	som = SOM.SOMNetwork(layer_dim, epochs=1)

	som.train(X)
	result = som.predict(X)

	print(result)
	print(result.shape)

	# plt.imshow(som.W)
	# plt.show()
	#hejhej

test()