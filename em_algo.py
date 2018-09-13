from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def data_gen(num_samples, num_clusters, dim = 2):

	data = np.zeros((num_samples, dim))

	means = np.zeros((num_clusters, dim))

	samples_per_cluster = num_samples//num_clusters

	plot_colors = ['ro', 'bo', 'go', 'co', 'yo', 'mo']

	horizontal = 5
	vertical = 5

	hor_step = 5
	ver_step = -5

	for i in range(len(means)):
		if (i % 2 == 0):
			#move horizontally
			means[i] = np.array([horizontal, vertical])
			horizontal += hor_step
		else:
			#move vertically
			means[i] = np.array([horizontal, vertical])
			vertical += (-1)*ver_step
			ver_step *= -1

	data_offset = 0
	for i, mean in enumerate(means):
		cov = np.random.random((dim, dim))
		data[data_offset:data_offset + samples_per_cluster] = np.random.multivariate_normal(mean, cov, samples_per_cluster)
		x_val = data[data_offset:data_offset + samples_per_cluster].T[0]
		y_val = data[data_offset:data_offset + samples_per_cluster].T[1]
		#plt.plot(x_val, y_val, plot_colors[i])
		data_offset += samples_per_cluster

	x_values = data.T[0]
	y_values = data.T[1]


	#plt.plot(x_values, y_values, 'ro')
	#plt.show()
	#plt.close()
	return data



def expectation_maximization(X):
	model = GaussianMixture()
	db = DBSCAN(eps=1, min_samples=10).fit(X)
	#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	#core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	#print(db.get_params())
	print("Number of clusters",len(set(labels)))
	print("Number of clusters without noise", n_clusters_)
	num_clusters = len(set(labels))
	for i in range(num_clusters):
		model.set_params(n_components=i+1)
		model.fit(X)
		#print(model.get_params())
		print(model.score(X))


X = data_gen(num_samples=100, num_clusters=5)
expectation_maximization(X)