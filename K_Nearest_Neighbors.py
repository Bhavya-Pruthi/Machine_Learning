# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 16:06:22 2019

@author: bhavya
"""
import random
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
dataset=pd.read_csv("Social_Network_Ads.csv")
dataset.drop(labels=["Gender"],axis=1,inplace=True)
a=dataset.iloc[:,-1].unique()
dataset=dataset.values.tolist()
random.shuffle(dataset)
test_size=0.2
training_data=dataset[:-int(0.2*len(dataset))]
test_data=dataset[-int(0.2*len(dataset)):]
training_set={}
test_set={}
for abc in a:
    training_set[abc]=[]
    test_set[abc]=[]
for i in training_data:
    training_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])  


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result
correct=0
total=0
for group in test_set:
    for d in test_set[group]:
        vote=k_nearest_neighbors(training_set,d,k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', correct/total)
