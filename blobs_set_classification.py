from matplotlib import pyplot as plt
import sklearn as sk
from sklearn import datasets, preprocessing, model_selection
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.patches as mpatches
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import os
import csv
import pandas as pd

#####################
# Software in charge of performing classification research
# on blobs dataset with use of combined classifiers
###

scores = np.zeros(2)

X, y = sk.datasets.make_blobs(n_samples = 5000, centers = 2)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

AdaBclf = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='linear'), n_estimators=10)
Baggingclf = BaggingClassifier(base_estimator=SVC(), n_estimators=10,random_state=0)

AdaBclf.fit(X_train, y_train)
scores[0] = accuracy_score(y_test, AdaBclf.predict(X_test)) *100
Baggingclf.fit(X_train, y_train)
scores[1] = accuracy_score(y_test, Baggingclf.predict(X_test)) *100

print('AdaB (in %): ', scores[0], '\n')
print('Bagging (in %): ', scores[1], '\n')






