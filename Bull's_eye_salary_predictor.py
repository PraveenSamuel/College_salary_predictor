#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 23:14:47 2018

@author: precillashaline.j
"""

# Bull's Eye Salary Predictor
# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('CollegeSalary.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
"""regressor.fit(X_train, y_train)"""
regressor.fit(X,y)


# Predicting the Test set results
y_pred = regressor.predict(8.4)
"""y_pred = regressor.predict(X_test)"""


# Visualising the Training set results
"""plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')"""

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
