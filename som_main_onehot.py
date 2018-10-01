import matplotlib
import matplotlib.pyplot as plt
import som as SOM
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import onehot_ecoding as one
import timeit


def main():
    df = one.read_data('student_data/student-mat.csv')
    B, values_per_column = one.raw_to_binary(df)
    R = one.binary_to_raw(B, values_per_column)
    one.validate_encoding(R, df)
    #print(R[2])
    #print(df.iloc[2, :])

    X = B
    print(X.shape)

    layer_dim = [12,12,X.shape[1]]
    som = SOM.SOMNetwork(layer_dim, epochs=500)

    start = timeit.default_timer()
    som.train(X)
    result = som.predict(X)
    stop = timeit.default_timer()

    print('Time: ', stop - start)  

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
	

main()
