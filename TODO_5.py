import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
    in : - X < [studentID, x, y] >
         - max_num_clusters
    out: - < [studentID, x, y, clusterID] >
"""
def kmeans_training(X, num_clusters):
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

def plot_clusters(student_per_cluster, num_clusters):
    colors = ['mo', 'go', 'bo', 'ro']
    for i in range(num_clusters):
        # Find which students are in cluster i
        index_curr_cluster = np.where(student_per_cluster.T[3] == i )[0]
        # Take the (x, y) coordinates for all students in cluster i
        x_values = student_per_cluster[index_curr_cluster].T[1]
        y_values = student_per_cluster[index_curr_cluster].T[2]
        plt.plot(x_values, y_values, colors[i])
    plt.show()
    
def main_TODO_5():
    # Toy data in order to test if elbow will return the 
    #   optimal number of clusters to be 4, as there are
    #   4 distributions of data.
    student_data1 = np.random.randint(0, 5,size=(20,3))
    student_data2 = np.random.randint(7, 12,size=(20,3))
    student_data3 = np.random.randint(14, 19,size=(20,3))
    student_data4 = np.random.randint(21, 26,size=(20,3))
    student_data1_2 = np.concatenate([student_data1, student_data2], axis=0)
    student_data1_2_3 = np.concatenate([student_data1_2, student_data3], axis=0)
    student_data = np.concatenate([student_data1_2_3, student_data4], axis=0)
    # Train data with the optimum number of clusters
    student_per_cluster = kmeans_training(student_data, 4)
    plot_clusters(student_per_cluster, 4)

main_TODO_5()
