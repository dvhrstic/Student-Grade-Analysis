#  Reads data from student-por.csv
#   for encoding it into binary values
import numpy as np
import pandas as pd
# Encoded matrix for storing one hot encoding

def read_data(file_name):
    return pd.read_csv(file_name, sep=";")

def raw_to_binary(df):
    titles = df.columns.values
    # Count the total length of binary for student
    num_bin_comb = 0
    for i, column in enumerate(df):
        unique_values = df[titles[i]].unique()
        num_bin_comb += len(unique_values)
    B = np.zeros( (len(df.values), num_bin_comb) , dtype=int)

    number_uniqvalues = np.zeros(len(titles), dtype=int)
    values_per_column = []
    # Iterate through each column
    matrix_binary_index = 0
    for i, column in enumerate(df):
        unique_values = df[titles[i]].unique()
        values_per_column.append(unique_values)
        number_uniqvalues[i] = len(unique_values)
        binary_data = []
        # Iterate through all the row values
        for j, row in enumerate(df.values):
            one_hot_results = np.zeros(number_uniqvalues[i], dtype=int)
            value = df.at[j, column]
            one_hot_results[np.where(unique_values == value)[0][0]] = 1
            binary_data.append(one_hot_results)
        # Insert the binary data into the final matrix
        binary_data = np.asarray(binary_data).T
        for k in range(len(binary_data)):
            B.T[:][matrix_binary_index] = binary_data[k]
            matrix_binary_index += 1   

    return B, values_per_column

def binary_to_raw(B, values_per_column):
    print(len(B[0]))
    R = np.zeros(B.shape,dtype=int)
    for i, row in enumerate(B):
        bin_matrix_index = 0
        for j, col in enumerate(values_per_column):
            
            print(values_per_column)
            value_location = np.where(B[i][bin_matrix_index: bin_matrix_index + len(values_per_column[j])] > 0)[0][0]
            column_value = values_per_column[j][value_location]
            R[i][j] = column_value    
            matrix_index += len(values_per_column[j])    
    return R        

def validate_encoding(R, df):
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
    

# df = read_data('student_data/student-por.csv')
# B, values_per_column = raw_to_binary(df)
# R = binary_to_raw(B, values_per_column)
# print(R[2])
# print(df.at[2,df.columns.values])
# validate_encoding(R, df)