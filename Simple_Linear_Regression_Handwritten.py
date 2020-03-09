# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:00:13 2018

@author: bhavya
"""
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
#To style as ggplot 
from matplotlib import style
style.use('ggplot')



data=pd.read_csv("Salary_Data.csv")

X = data.iloc[:,0].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Random Values
weight=1
bias=0

def predict(x, weight, bias):
    return weight*x + bias

def cost_function(X_train, Y_train, weight, bias):
    len_of_data = len(X_train)
    total_error = 0.0
    for i in range(len(len_of_data)):
        total_error += (Y_train[i] - (weight*X_train[i] + bias))**2
    return total_error / len_of_data

def update_weights(X_train, Y_train, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    len_of_data = len(X_train)

    for i in range(len_of_data):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv += -2*X_train[i] * (Y_train[i] - (weight*X_train[i] + bias))

        # -2(y - (mx + b))
        bias_deriv += -2*(Y_train[i] - (weight*X_train[i] + bias))

    weight -= (weight_deriv / len_of_data) * learning_rate
    bias -= (bias_deriv / len_of_data) * learning_rate

    return weight, bias

def train(radio, sales, weight, bias, learning_rate, iters):
    cost_history = []

    for i in range(iters):
        weight,bias = update_weights(radio, sales, weight, bias, learning_rate)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weight, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print "iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost)

    return weight, bias, cost_history

plt.scatter(X_train,y_train,color='blue',label='data')
plt.plot(X_train, regression_line, label='regression line')
plt.legend(loc=2)
plt.show()

#To Make Prediction

y_pred=[predict(x) for x in X_test]


plt.scatter(X_train,y_train,color='blue',label='data')
plt.scatter(X_test,y_test,color="black")
plt.plot(X_train, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()




