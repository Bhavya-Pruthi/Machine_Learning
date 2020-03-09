# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:05:27 2019

@author: bhavya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#To style as ggplot 
from matplotlib import style
style.use('ggplot')

data=pd.read_csv("50_Startups.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

#Convert States into numeric values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Remove the Dummy Variable 
X = X[:, 1:]


#To check Coorelation
import statsmodels.formula.api as sm
X_opt=X[:,:]
reg=sm.OLS(endog=y,exog=X_opt).fit()
reg.summary()

X=X[:,[-1,-2]]

m=len(y)   
x0 = np.ones(m)
X=np.c_[x0,X]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train= sc_y.fit_transform(y_train.reshape(-1, 1))
y_train=y_train.reshape(-1, )
#shape of B=(no of features+1,1)
B = np.array([0,0,0])

alpha = 0.005
#To calculate Cost Function 
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X_train, y_train, B)
print(inital_cost)

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

final_B, cost_history = gradient_descent(X_train, y_train, B, alpha, 100000)
print(final_B)
print(cost_history[-1])

#To Make Prediction
Y_pred = X_test.dot(final_B)
Y_pred=sc_y.inverse_transform(Y_pred)

#To plot one column
x = np.linspace(min(X_train[:,1]),max(X_train[:,1]),100)
y = final_B[0] + final_B[1] * x +final_B[2] * x 
plt.scatter(X_train[:,1],y_train,color='blue',label='data')
#plt.scatter(X_train[:,2],y_train,color='red',label='data')
plt.scatter(X_test[:,1],sc_y.fit_transform(y_test.reshape(-1, 1)),color="black")
plt.plot(x,y, label='regression line')
plt.legend(loc=4)
plt.show()

error=sum(Y_pred)-sum(y_test)


