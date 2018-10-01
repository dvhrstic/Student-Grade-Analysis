import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import som as SOM
import model as Mod
import seaborn as sns; sns.set()
import onehot_ecoding as one
import sys

def driver(user):
	if (user == 1):
		f = open("Student_data/student-mat.bin","rb")
		X = np.load(f)

	else:
		df = one.read_data('student_data/student-mat.csv')
		X, values_per_column = one.raw_to_binary(df)
		R = one.binary_to_raw(X, values_per_column)
		one.validate_encoding(R, df)

	# Remove the columns with grades
	X = X[:, :-3]
	layer_dim = [12,12,X.shape[1]]

	# som = SOM.SOMNetwork(layer_dim, epochs=500)
	mod = Mod.Model()

	mod.reduce_dim(X, [12,12], 500)

	f = open("Student_data/student2D.bin","rb")
	# [studentID, x, y]
	student_2d = np.load(f)
	# print(student_2d.shape)
	# mod.plot_student2D(student_2d)

	mod.elbow_method(student_2d, 10)
	
	opt_clusters = int(input('Optimal number of cluster acc to the graph: '))
	plt.close()

	# [studentID, x, y, clsuterID]
	student_cluster_data = mod.kmeans_training(student_2d, opt_clusters)
	mod.plot_clusters(student_cluster_data, opt_clusters)

	mod.save_clusters_csv(student_cluster_data, opt_clusters, 'student-mat.csv')
	mod.plot_grades(opt_clusters)

if __name__ == '__main__':
	driver(int(sys.argv[1]))
