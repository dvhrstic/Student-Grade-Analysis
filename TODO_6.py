import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(student_data, num_clusters):
	"""Plot the clsuters in different colours
		----------
		student_data: array [studentID, x, y, clusterID]
			the input data
		num_clusters: int
			the number of clusters
	"""
	colors = [
		'#FF0000', '#FF8000',
		'#FFFF00', '#80FF00',
		'#00FF80', '#00FFFF',
		'#0000FF', '#7F00FF',
		'#FF00FF', '#FF007F',
		'#808080', '#000000',
		'#666600', '#994C00'
	]
	#sort the data by cluster
	# student_data = student_data[student_data[:,3].argsort()]

	plt.title('2D representation of the student data')
	for i in range(num_clusters):
		cluster = student_data[student_data[:,3] == i]
		x_pos = cluster.T[1]
		y_pos = cluster.T[2]
		plt.plot(x_pos, y_pos, colors[i],linestyle='None',marker='o', label='Cluster ' + str(i + 1))

	plt.legend(loc='upper left', ncol='2')
	plt.savefig('Plots/clustered_data.png')


# Testing data
a = np.array([
	[0,1,1,0],
	[1,2,2,1],
	[2,3,3,2],
	[3,4,4,3],
	[4,1.5,1.5,0],
	[5,2.5,2.5,1],
	[6,3.5,3.5,2],
	[7,4.5,4.5,3]
	])

plot_clusters(a, 4)