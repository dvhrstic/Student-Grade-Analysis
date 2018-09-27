import numpy as np
import pickle
import pandas as pd

def save_clusters_csv(student_data, num_clusters, file):
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

# TESTING DATA [studentID, x, y, clusterID]
student_test_data = np.array([
	[0,1,1,0],
	[1,2,2,1],
	[2,3,3,2],
	[3,4,4,3],
	[4,1.5,1.5,0],
	[5,2.5,2.5,1],
	[6,3.5,3.5,2],
	[7,4.5,4.5,3]
	],dtype='int')

save_clusters_csv(student_test_data, 4, 'student-mat.csv')
