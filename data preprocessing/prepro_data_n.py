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

	# decode_numeric(new_data, labels, label_map)
	file_name = file[:-4]
	f = open("Student_data/" + file_name + ".bin","wb")
	np.save(f, new_data)


	return new_data

def decode_numeric(data, labels, label_map, print_data=False, save_data=False, filename=''):

	if(print_data):
		for i in range(len(labels)):
			print(labels[i], end='\t')

		print()

		for i in range(len(data)):
			for j in range(len(labels)):
				print(label_map[labels[j]][data[i][j]], end='\t')
			print() 
	else:
		df_data = pd.DataFrame(index=range(0,len(data)), columns=labels)
		for i in range(len(data)):
			for j in range(len(labels)):
				df_data.at[i, labels[j]] = label_map[labels[j]][data[i][j]]

		if(save_data):
			df_data.to_csv(path_or_buf='data/' + filename , sep=';')
		else:		
			print(df_data)

read_data('student-por.csv')


