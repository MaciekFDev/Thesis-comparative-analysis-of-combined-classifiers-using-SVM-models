from matplotlib import pyplot as plt
import sklearn as sk
from sklearn import datasets, preprocessing, model_selection
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score
import numpy as np
import matplotlib.patches as mpatches
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import os
import csv
import pandas as pd

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#####################
# Software in charge of performing classification research
# on cirle dataset with use of combined classifiers
###

scores = np.zeros(2)
scores_adab = []
scores_bagging = []
scores_f1_adab = []
scores_f1_bagging = []

#X, y = sk.datasets.make_gaussian_quantiles(n_samples = 5000, n_classes = 2)

# Loading datasets
X = np.loadtxt('C:/Users/Veteran/Thesis-combined-svm-algorithms-analysis/'
                'X_circle_file5.txt', delimiter = ',')
y = np.loadtxt('C:/Users/Veteran/Thesis-combined-svm-algorithms-analysis/'
                'y_circle_file5.txt', delimiter = ',')

# Splitting it into training and testing parts
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

i = 10

# Creating classifiers ensemblies
AdaBclf = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='poly'),
                            n_estimators=i)
Baggingclf = BaggingClassifier(base_estimator=SVC(probability=True, kernel='poly'),
                            n_estimators=i)

# Training and calculating accuracy score for reach ensemble
AdaBclf.fit(X_train, y_train)
scores_adab.append(accuracy_score(y_test, AdaBclf.predict(X_test)) *100)
scores_f1_adab.append(f1_score(y_test, AdaBclf.predict(X_test), average = 'weighted') *100)
Baggingclf.fit(X_train, y_train)
scores_bagging.append(accuracy_score(y_test, Baggingclf.predict(X_test)) *100)
scores_f1_bagging.append(f1_score(y_test, Baggingclf.predict(X_test), average = 'weighted') *100)


print('AdaBoost accuracy score: ', scores_adab, '\n')
print('Bagging accuracy score: ', scores_bagging, '\n')
print('AdaBoost f1 score: ', scores_f1_adab, '\n')
print('Bagging f1 score: ', scores_f1_bagging, '\n')

# x_range = range(5,61,5)
# plt.figure(1)
# plt.plot(x_range, scores_adab, marker='o', label='Accuracy score for AdaBoost')
# plt.plot(x_range, scores_bagging, marker="x", label='Accuracy score for Bagging')
# plt.legend()
# plt.show()

# plt.figure(2)
# plt.plot(x_range, scores_f1_adab, marker='o', label='F1 score for AdaBoost')
# plt.plot(x_range, scores_f1_bagging, marker="x", label='F1 score for Bagging')
# plt.legend()
# plt.show()

# Discrete part
# Drawing confusion matrix
plot_confusion_matrix(AdaBclf, X_test, y_test)
plt.show()

plot_confusion_matrix(Baggingclf, X_test, y_test)
plt.show()

# Drawing decision surface for each classifier.
fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface for AdaBoost ensemble method')
# Set-up grid for plotting.
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, AdaBclf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('Feature 1')
ax.set_xlabel('Feature 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()

fig1, ax1 = plt.subplots()
# title for the plots
title = ('Decision surface of Bagging ensemble method ')
# Set-up grid for plotting.
X0, X1 = X_test[:, 0], X_test[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax1, Baggingclf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax1.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax1.set_ylabel('Feature 1')
ax1.set_xlabel('Feature 2')
ax1.set_xticks(())
ax1.set_yticks(())
ax1.set_title(title)
ax1.legend()
plt.show()