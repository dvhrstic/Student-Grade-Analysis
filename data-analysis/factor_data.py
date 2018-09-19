import matplotlib.pyplot as plt
import numpy as np

c1 = np.random.choice([0,1,2,3,4], size=100, p=[0.1, 0.1, 0.3, 0.2, 0.3])
c2 = np.random.choice([0,1,2,3,4], size=100, p=[0.2, 0.1, 0.4, 0.2, 0.1])
c3 = np.random.choice([0,1,2,3,4], size=100, p=[0.2, 0.4, 0.2, 0.1, 0.1])
c4 = np.random.choice([0,1,2,3,4], size=100, p=[0.1, 0.2, 0.5, 0.1, 0.1])
c5 = np.random.choice([0,1,2,3,4], size=100, p=[0.2, 0.3, 0.4, 0.1, 0.0])
c6 = np.random.choice([0,1,2,3,4], size=100, p=[0.0, 0.1, 0.6, 0.2, 0.1])

# Assign colors for each airline and the names
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00', '#D57800']
names = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6']
         
# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([c1, c2, c3, c4, c5, c6], bins = [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25], stacked=True,
         color = colors, label=names)

# Plot formatting
plt.legend(ncol=1)
#plt.ylim((0,200))
plt.xlabel('Education level')
plt.ylabel('Number of students')
plt.xticks(np.arange(5), ('None', 'Primary', '5th-9th grade ', 'Secondary', 'Higher'))
plt.title("Stacked histogram for mothers' education level")
plt.show()