import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
import model as Mod
import seaborn as sns; sns.set()
import dataPreprocessing.onehot_ecoding as one
import sys

def driver(user, file):
	if (file == 1):
		file_name = "student-por"
	else:
		file_name = "student-mat"

	if (user == 1):
		f = open("student_data/"+ file_name +".bin","rb")
		X = np.load(f)
	else:
		df = one.read_data('student_data/'+ file_name +'.csv')
		X, values_per_column = one.raw_to_binary(df)
		R = one.binary_to_raw(X, values_per_column)
		one.validate_encoding(R, df)

	# Remove the columns with grades
	X = X[:, :-3]
	# Familly related
	#X = np.concatenate((X[:,4:12], X[:,17:18]), axis=1)

	mod = Mod.Model()

	mod.reduce_dim(X, [20,20], 1000, file_name)

	f = open("student_data/"+ file_name +"/student2D.bin","rb")
	# [studentID, x, y]
	student_2d = np.load(f)

	mod.elbow_method(student_2d, 10)
	
	opt_clusters = int(input('Optimal number of cluster acc to the graph: '))
	plt.close()

	# [studentID, x, y, clsuterID]
	student_cluster_data = mod.kmeans_training(student_2d, opt_clusters)
	mod.plot_clusters(student_cluster_data, opt_clusters, file_name)

	mod.save_clusters_csv(student_cluster_data, opt_clusters, file_name)
	mod.plot_grades(opt_clusters, file_name)
	mod.get_variance(opt_clusters, file_name, 30)

if __name__ == '__main__':
	driver(int(sys.argv[1]), int(sys.argv[2]))
