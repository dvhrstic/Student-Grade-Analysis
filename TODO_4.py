from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

"""
    in : - student_data < [studentID, x, y] >
         - max_num_clusters
    out: - output scores for different #clusters
"""

def elbow_method(X, max_num_clusters):
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

def main_TODO_4():
    # Toy data in order to test if elbow will return the 
    #   optimal number of clusters to be 4, as there are
    #   4 distributions of data.
    student_data1 = np.random.randint(0, 10,size=(200,3))
    student_data2 = np.random.randint(20, 30,size=(200,3))
    student_data3 = np.random.randint(40, 50,size=(200,3))
    student_data4 = np.random.randint(60, 70,size=(200,3))
    student_data1_2 = np.concatenate([student_data1, student_data2], axis=0)
    student_data1_2_3 = np.concatenate([student_data1_2, student_data3], axis=0)
    student_data = np.concatenate([student_data1_2_3, student_data4], axis=0)
    elbow_method(student_data, 10)
main_TODO_4()