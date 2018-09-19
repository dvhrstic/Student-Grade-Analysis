import matplotlib.pyplot as plt
import numpy as np

# data = np.random.randint(1, 7, (100,))

# n, bins, patches = plt.hist(data, alpha=0.5, align="mid")

g1 = [5, 15, 5, 20, 30, 25]
g2 = [25, 30, 20, 5, 15, 5]
g3 = [15, 15, 20, 20, 15, 15]
g4 = [5, 5, 20, 30, 30, 10]
g5 = [2, 28, 30, 25, 12, 3]
g6 = [10, 15, 25, 25, 15, 10]

grades = [g1, g2, g3, g4, g5, g6]

# red, orange, dark yellow, bright yellow, green-yellow, green
colors = ['#FF3333', '#FF8033', '#FFC133', '#FCFF33', '#BEFF33', '#42FF33']

x = np.arange(6)

for i, grade in enumerate(grades):
	plt.title('The distribution of grades for cluster ' + str(i))
	plt.xlabel('Grade')
	plt.ylabel('Number of students')
	plt.bar(x, grade, color=colors)
	plt.xticks(x, ('F', 'E', 'D', 'C', 'B', 'A'))
	plt.savefig('grades_cluster_' + str(i) + '.png')
	plt.close()