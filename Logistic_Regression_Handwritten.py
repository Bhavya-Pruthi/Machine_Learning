# -*- coding: utf-8 -*-
"""
Created on Thu Feb 6 16:48:30 2019

@author: bhavya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose


    def _add_intercept(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]     
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self._sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
    
        return self._sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

    
    def preprocessing_data(self,X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train,X_test,y_train,y_test
    def visualize_data(self,X_set, y_set):   
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        mesh=np.array([X1.ravel(), X2.ravel()]).T
        mesh=self._add_intercept(mesh)
        plt.contourf(X1, X2, self.predict(mesh,0.5).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#        plt.contourf(X1, X2, self.predict(np.array([X1.ravel(), X2.ravel()]).T,0.5).reshape(X1.shape),
#                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        plt.scatter(X_set[y_set == 0][:, 1], X_set[y_set == 0][:, 2], color='r', label='0')
        plt.scatter(X_set[y_set == 1][:, 1], X_set[y_set == 1][:, 2], color='g', label='1')
        plt.title('Logistic Regression (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
    

model = LogisticRegression(lr=0.01, num_iter=1000)

X_train, X_test, y_train, y_test=model.preprocessing_data(X)

model.fit(X_train,y_train)

preds = model.predict(X_test,0.5)

(preds == y_test).mean()

model.visualize_data(X_test,y_test)


    





