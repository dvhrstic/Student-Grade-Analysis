import numpy as np
import pickle
import random as rd
import pandas as pd

def read_data(file):
	directory = 'student_data/'
	file = file

	path = directory + file

	# read the csv as a Pandas df
	df_students = pd.read_csv(filepath_or_buffer = path, sep = ';')
	
	# get all the labels
	labels = df_students.columns.values
	# create a dictionary for mapping the labels to numeric values
	label_map = dict()

	# get the options for each of the labels and connect
	# it to its label ('label' => array(option1, option2, ...))
	for label in labels:
		label_options = np.sort(df_students[label].unique())
		label_map[label] = label_options

	# create a numpy array to hold the numeric representation of the data
	new_data = np.zeros(df_students.shape, dtype=int)

	for index, student in df_students.iterrows():
		for i, col in enumerate(labels):
			new_data[index][i] = np.where(label_map[col] == student[col])[0]

	return new_data

def decode_numeric(data, labels, labels_map):

	for i in range(len(labels)):
		print(labels[i])
		print(label_map[labels[i]][new_data[0][i]]) 


read_data('student-mat.csv')


