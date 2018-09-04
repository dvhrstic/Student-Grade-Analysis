#  Reads data from student-por.csv
#   for encoding it into binary values
import csv
import numpy as np
import pandas as pd

df = pd.read_csv('student_data/student-por.csv', sep=";")

id2title = {};
title2id = {};
titles = df.columns.values;
number_uniqvalues = np.zeros(len(titles), dtype=int);
number_students = len(df.values);

# Encoded matrix for storing one hot encoding
H = [];

for i, column in enumerate(df):
    id2title[i] = titles[i];
    title2id[id2title[i]] = i;
    unique_values = df[id2title[i]].unique();
    number_uniqvalues[i] = len(unique_values);
    binary_data = [];
    for j, row in enumerate(df.values):
        one_hot_results = np.zeros(number_uniqvalues[i], dtype=int);
        value = df.at[j, column]
        one_hot_results[np.where(unique_values == value)[0][0]] = 1;
        binary_data.append(one_hot_results)

    H.append(binary_data)

