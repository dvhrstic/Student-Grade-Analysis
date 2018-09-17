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



def expectation_maximization_DBSCAN(X):
	db = DBSCAN(eps=0.5, min_samples=50).fit(X)
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print("---DBSCAN---")
	print("---Number of clusters: ",len(set(labels)))
	print("---Number of clusters without noise: ", n_clusters_)


def elbow_method(X, model, num_clusters):
	k_clusters = np.arange(1, num_clusters)
	scores = []
	for k in k_clusters:
		model.set_params(n_components=k)
		model.fit(X)
		scores.append(model.score(X))
	plt.plot(k_clusters, scores)
	plt.title('Elbow Method')
	plt.xlabel('Number of clusters')
	plt.ylabel('Scores')
	plt.show()

def main():
	# Data generation
	X = data_gen(num_samples=100, num_clusters=5)	
	# EM Model initialization
	model = GaussianMixture()
	# Getting the optimal number of clusters with DBSCAN method
	expectation_maximization_DBSCAN(X)
	# Elbow method for different number of clusters
	elbow_method(X,model, num_clusters=10)


main()