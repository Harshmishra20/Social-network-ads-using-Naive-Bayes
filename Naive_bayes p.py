# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:16:19 2023

@author: Dell
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\Data Science\Daily Practice\April\03-04-2023\Social_Network_Ads.csv")

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
X_train1=sc1.fit_transform(X_train) 
X_test1=sc1.transform(X_test)

from sklearn.naive_bayes import BernoulliNB
classifier1=BernoulliNB()
classifier1.fit(X_train,y_train)

y_pred1=classifier1.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_test, y_pred1)
print(ac1)

bias1 = classifier1.score(X_train, y_train)
print(bias1)
variance1 = classifier1.score(X_test, y_test)
print(variance1)

from sklearn.naive_bayes import GaussianNB
classifier2=GaussianNB()
classifier2.fit(X_train1,y_train)

y_pred2=classifier2.predict(X_test1)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

from sklearn.metrics import accuracy_score
ac2 = accuracy_score(y_test, y_pred2)
print(ac2)

bias2 = classifier2.score(X_train1, y_train)
print(bias2)

variance2 = classifier2.score(X_test1, y_test)
print(variance2)



from sklearn.preprocessing import Normalizer
NM=Normalizer()
X_train2=NM.fit_transform(X_train) 
X_test2=NM.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
classifier3=MultinomialNB()
classifier3.fit(X_train2,y_train)

y_pred3=classifier3.predict(X_test2)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)

from sklearn.metrics import accuracy_score
ac3 = accuracy_score(y_test, y_pred3)
print(ac3)

bias3 = classifier3.score(X_train2, y_train)
print(bias3)
variance3 = classifier3.score(X_test2, y_test)
print(variance3)