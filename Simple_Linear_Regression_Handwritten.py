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

#Returns Slope of Best Fit Line
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys,ys_line):
    y_mean_line = [mean(ys) for y in ys]
    squared_error_regr = squared_error(ys, ys_line)
    squared_error_y_mean = squared_error(ys, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
    
m, b = best_fit_slope_and_intercept(X_train,y_train)
regression_line = [(m*x)+b for x in X_train]

r_squared = coefficient_of_determination(y_train,regression_line)
print(r_squared)



plt.scatter(X_train,y_train,color='blue',label='data')
plt.plot(X_train, regression_line, label='regression line')
plt.legend(loc=2)
plt.show()

#To Make Prediction

y_pred=[(m*x)+b for x in X_test]


plt.scatter(X_train,y_train,color='blue',label='data')
plt.scatter(X_test,y_test,color="black")
plt.plot(X_train, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()




