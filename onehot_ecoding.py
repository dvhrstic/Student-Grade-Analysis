#  Reads data from student-por.csv
#   for encoding it into binary values
import numpy as np
import pandas as pd
# Encoded matrix for storing one hot encoding

def read_data(file_name):
    return pd.read_csv(file_name, sep=";")

def raw_to_binary(df):
    number_students = df.shape[0]
    B = []
    id2title = {}
    title2id = {}
    titles = df.columns.values
    number_uniqvalues = np.zeros(len(titles), dtype=int)
    values_per_column = []
    for i, column in enumerate(df):
        id2title[i] = titles[i]
        title2id[id2title[i]] = i
        unique_values = df[id2title[i]].unique()
        values_per_column.append(unique_values)
        number_uniqvalues[i] = len(unique_values)
        binary_data = []
        for j, row in enumerate(df.values):
            one_hot_results = np.zeros(number_uniqvalues[i], dtype=int)
            value = df.at[j, column]
            one_hot_results[np.where(unique_values == value)[0][0]] = 1
            binary_data.append(one_hot_results)
        B.append(binary_data)
    return B, values_per_column

def binary_to_raw(B, values_per_column):
    R = np.zeros((len(B[0]), len(B)),dtype=object)
    B = np.array(B).transpose()
    for i, row in enumerate(B):
        values = []
        for j, col in enumerate(B[i]):
            values_per_column[j]
            value_location = np.where(B[i][j] > 0)[0][0]
            column_value = values_per_column[j][value_location]
            R[i][j] = column_value        
    return R        

def validate_encoding():
    B, values_per_column = raw_to_binary(df)
    R = binary_to_raw(B, values_per_column)
    err_count = 0
    for i, row in enumerate(R):
        for j,col in enumerate(R[i]):
            matrix_value = R[i][j]
            origin_value = df.at[i,df.columns.values[j]]
        if  matrix_value != origin_value: 
            print(matrix_value)
            print(origin_value)
            err_count = err_count + 1
    if err_count > 0:
        print("Encoding errorneous") 
    else:
        print("Encoding successful")
    

df = read_data('student_data/student-por.csv')
validate_encoding()