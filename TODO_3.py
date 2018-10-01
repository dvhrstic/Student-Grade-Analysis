import numpy as np
import matplotlib.pyplot as plt

"""
    in : student_data < [studentID, x, y] >
    out : Graph of low-dimensional student data
"""
def plot_student2D(student_data):
    plt.title(" 2 dimensional representation of student data")
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.plot(student_data.T[1], student_data.T[2], 'ro')
    plt.show()

def main_TODO_3():
    toy_student_data = np.random.randint(0, 200,size=(200,3))
    plot_student2D(toy_student_data)
main_TODO_3()
