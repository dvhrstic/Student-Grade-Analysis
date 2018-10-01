import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()

class Model():

	def reduce_dim(self, data, grid_size, epochs):
		"""Reduce the input to 2D using SOM
			Parameters
			----------
			data: n x m (numpy) array
				the input data
			grid_size: array [n_row, n_col]
				the shape of the output grid
		"""
		layer_dim = [grid_size[0],grid_size[1], data.shape[1]]

		som = SOM.SOMNetwork(layer_dim, epochs=epochs)
		som.train(data)

		result = som.predict(data)

		grid = self.create_grid(result, layer_dim)

		ax = sns.heatmap(grid, annot=False, fmt="d")
		plt.savefig("Plots/heatmap_noNum.png")
		plt.close()

		ax = sns.heatmap(grid, annot=True, fmt="d")
		plt.savefig("Plots/heatmap_withNum.png")
		plt.close()

		f = open("Student_data/student2D.bin","wb")
		np.save(f, result)

	def plot_student2D(self, student_data):
		"""
		in : student_data < [studentID, x, y] >
		out : Graph of low-dimensional student data
		"""
		plt.title(" 2 dimensional representation of student data")
		plt.xlabel("x - axis")
		plt.ylabel("y - axis")
		plt.plot(student_data.T[1], student_data.T[2], 'ro')
		plt.show()

	def elbow_method(self, X, max_num_clusters):
		"""
		in : - X < [studentID, x, y] >
		     - max_num_clusters
		out: - output scores for different #clusters
		"""

		# Remove the studentID column
		X = X.T[1:].T
		model = KMeans()
		k_clusters = np.arange(1, max_num_clusters)
		scores = []
		for k in k_clusters:
			model.set_params(n_clusters=k)
			model.fit(X)
			scores.append(model.score(X))
		plt.plot(k_clusters, scores)
		plt.title('Elbow Method')
		plt.xlabel('Number of clusters')
		plt.ylabel('Scores')
		plt.show()

	def kmeans_training(self, X, num_clusters):
		"""
		in : - X < [studentID, x, y] >
		     - max_num_clusters
		out: - < [studentID, x, y, clusterID] >
		"""
		model = KMeans()
		model.set_params(n_clusters=num_clusters)
		# No need for studentID column during training
		model.fit(X.T[1:].T)
		students_clusters = model.labels_
		# Add a final column with students_clusters labels
		output = np.zeros((len(X),len(X[0]) + 1))
		output[:,:-1] = X
		output.T[-1] = students_clusters
		return output

	# def plot_clusters(self, student_per_cluster, num_clusters):
	# 	colors = ['mo', 'go', 'bo', 'ro']
	# 	for i in range(num_clusters):
	# 		# Find which students are in cluster i
	# 		index_curr_cluster = np.where(student_per_cluster.T[3] == i )[0]
	# 		# Take the (x, y) coordinates for all students in cluster i
	# 		x_values = student_per_cluster[index_curr_cluster].T[1]
	# 		y_values = student_per_cluster[index_curr_cluster].T[2]
	# 		plt.plot(x_values, y_values, colors[i])
	# 	plt.show()

	def plot_clusters(self, student_data, num_clusters):
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

		ax = plt.subplot(111)
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.savefig('Plots/clustered_data.png')
		plt.close()

	def save_clusters_csv(self, student_data, num_clusters, file):
		"""Save the student data cluster wise for further analysis
			----------
			student_data: array [studentID, x, y, clusterID]
				the input data
			num_clusters: int
				the number of clusters
			file: string
				name of the file (either student-mat or student-por in this case)
		"""
		directory = 'student_data/'
		file = file

		path = directory + file

		# read the csv as a Pandas df
		df_students = pd.read_csv(filepath_or_buffer = path, sep = ';')

		for i in range(num_clusters):
			cluster = student_data[student_data[:,3] == i]
			df_stud_clust = df_students.loc[cluster[:,0]]
			df_stud_clust.to_csv(path_or_buf='student_data/students-cluster-' + str(i + 1) + '.csv', sep=';')

	def create_grid(self, data, dim):
		rows = dim[0]
		cols = dim[1]

		grid = np.zeros((rows, cols), dtype='int')

		for d in data:
			grid[d[1]][d[2]] += 1

		return grid
	def plot_grades(self, number_clusters, filename):

		grade_intervals = [[0, 3], [4, 9], [10, 13] , [14, 15], [16, 17], [18, 20]]
		# red, orange, dark yellow, bright yellow, green-yellow, green
		colors = ['#FF3333', '#FF8033', '#FFC133', '#FCFF33', '#BEFF33', '#42FF33']
		x = np.arange(6)

		tot_student_data = pd.read_csv(filename, sep=";")
		tot_student_grades = np.asarray(tot_student_data["G3"])
		tot_grades = np.zeros(len(grade_intervals), dtype=int)
		for j,inter in enumerate(grade_intervals):
			tot_grades[j] = len(np.where((tot_student_grades >= inter[0]) & \
						   (tot_student_grades <= inter[1])) [0])

		plt.title('The distribution of grades for all students')
		plt.xlabel('Grades')
		plt.ylabel('Number of students')
		plt.bar(x, tot_grades, color=colors)
		plt.xticks(x, ('Poor', 'Weak', 'Sufficient', 'Good', 'Very Good', 'Excellent'))
		plt.savefig("Plots/tot_grades.png")
		plt.close()

		for i in range(number_clusters):
			file_name = "student_data/students-cluster-" + str(i+1) + ".csv"
			
			student_data = pd.read_csv(file_name, sep=";")
			student_grades = np.asarray(student_data["G3"])
			print("num students in cluster ", i, " : ", len(student_grades))
			grades = np.zeros(len(grade_intervals), dtype=int)
			for j,inter in enumerate(grade_intervals):
				grades[j] = len(np.where((student_grades >= inter[0]) & \
							   (student_grades <= inter[1])) [0])

			plt.title('The distribution of grades for cluster ' + str(i + 1))
			plt.xlabel('Grades')
			plt.ylabel('Percentage of students')
			plt.bar(x, (grades/tot_grades)*100, color=colors)
			plt.xticks(x, ('Poor', 'Weak', 'Sufficient', 'Good', 'Very Good', 'Excellent'))
			x1,x2,y1,y2 = plt.axis()
			plt.axis((x1,x2,0,100))
			plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
			plt.savefig('Plots/grades_cluster_' + str(i + 1) + '.png')
			plt.close()


		

		

