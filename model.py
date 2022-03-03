# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:57:40 2022

@author: alokr
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset= pd.read_csv('hiring.csv')

#feature engineering
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# creating all independent variable'

# Python iloc() function enables us to select a particular cell of the 
#dataset, that is, it helps us select a value that belongs to a particular
# row or column from a set of values of a data frame or dataset.
x= dataset.iloc[:, :3]

# converting words to integer values (writing function)
def convert_to_int(word):
    word_dict={'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 
               'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11,
               'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

x['experience']= x['experience'].apply(lambda x : convert_to_int(x))

# for dependent varaible

y= dataset.iloc[:,-1]

# splitting training and test set
# since we have very small dataset, we will train our model with all available dataset

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()

# fitting model with training data
regressor.fit(x,y)

# saving model to disk

pickle.dump(regressor, open('model.pkl','wb'))

#Loading model to compare results
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))






