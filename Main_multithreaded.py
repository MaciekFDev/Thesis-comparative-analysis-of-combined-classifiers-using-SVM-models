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
import threading

class myThread (threading.Thread):
    def __init__(self, threadID, name, counter, clf, X, y):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.clf = clf
        self.X = X
        self.y = y
    
    def run(self):
        threadLock.acquire()
        self.clf.fit(self.X, self.y)
        threadLock.release()

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
AdaBclf = []
Baggingclf = []
scores_adab = []
scores_bagging = []
scores_f1_adab = []
scores_f1_bagging = []

#X, y = sk.datasets.make_gaussian_quantiles(n_samples = 5000, n_classes = 2)

print(os.getpid())
# Loading datasets
X = np.loadtxt('C:/Users/Veteran/Thesis-combined-svm-algorithms-analysis/'
                'X_class_file5.txt', delimiter = ',')
y = np.loadtxt('C:/Users/Veteran/Thesis-combined-svm-algorithms-analysis/'
                'y_class_file5.txt', delimiter = ',')

# Splitting it into training and testing parts
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

for i in range(5, 100, 5):
    print(i)
    # Creating classifiers ensemblies
    AdaBclf.append(AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='poly'),
                            n_estimators=i))
    Baggingclf.append(BaggingClassifier(base_estimator=SVC(probability=True, kernel='poly'),
                            n_estimators=i))

threadLock = threading.Lock()
threads = []

for i in range(19):
    # Training and calculating accuracy score for reach ensemble within individual thread
    thread1 = threading.Thread(target = AdaBclf[i].fit, args=(X_train, y_train), name = 'Thread '+str(i))
    thread2 = threading.Thread(target = Baggingclf[i].fit, args=(X_train, y_train), name = 'Thread '+str(i+20))
    #thread1 = myThread(i, 'Thread '+str(i), i, AdaBclf[i], X_train, y_train)
    #thread2 = myThread(i+20, 'Thread '+str(i+20), i+20, Baggingclf[i], X_train, y_train)
    #thread1 = threading.start_new_thread(AdaBclf[i].fit, (X_train, y_train))
    #thread2 = threading.start_new_thread(Baggingclf[i].fit, (X_train, y_train))
    threads.append(thread1)
    threads.append(thread2)

for t in threads:
    t.start()

for t in threads:
    print(t.name, t.is_alive())

print('stworzono wateczki')
# Waiting for all threads to finish
for t in threads:
    t.join()
    print(t.name, t.is_alive())

print('poczekano na wateczki')
# Predicting score for each trained classifier
for i in range(19):
    print(i)
    scores_adab.append(accuracy_score(y_test, AdaBclf[i].predict(X_test)) *100)
    scores_f1_adab.append(f1_score(y_test, AdaBclf[i].predict(X_test), average = 'weighted') *100)

    scores_bagging.append(accuracy_score(y_test, Baggingclf[i].predict(X_test)) *100)
    scores_f1_bagging.append(f1_score(y_test, Baggingclf[i].predict(X_test), average = 'weighted') *100)

print('AdaBoost accuracy score: ', scores_adab, '\n')
print('Bagging accuracy score: ', scores_bagging, '\n')
print('AdaBoost f1 score: ', scores_f1_adab, '\n')
print('Bagging f1 score: ', scores_f1_bagging, '\n')
x_range = range(5,106,5)
plt.figure(1)
plt.plot(x_range, scores_adab, marker='o', label='Accuracy score for AdaBoost')
plt.plot(x_range, scores_bagging, marker="x", label='Accuracy score for Bagging')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x_range, scores_f1_adab, marker='o', label='Accuracy score for AdaBoost')
plt.plot(x_range, scores_f1_bagging, marker="x", label='Accuracy score for Bagging')
plt.legend()
plt.show()

# # Discrete part
# # Drawing confusion matrix
# plot_confusion_matrix(AdaBclf, X_test, y_test)
# plt.show()

# plot_confusion_matrix(Baggingclf, X_test, y_test)
# plt.show()

# # Drawing decision surface for each classifier.
# fig, ax = plt.subplots()
# # title for the plots
# title = ('Decision surface for AdaBoost ensemble method')
# # Set-up grid for plotting.
# X0, X1 = X_test[:, 0], X_test[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# plot_contours(ax, AdaBclf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel('Feature 1')
# ax.set_xlabel('Feature 2')
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)
# ax.legend()
# plt.show()

# fig1, ax1 = plt.subplots()
# # title for the plots
# title = ('Decision surface of Bagging ensemble method ')
# # Set-up grid for plotting.
# X0, X1 = X_test[:, 0], X_test[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# plot_contours(ax1, Baggingclf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax1.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax1.set_ylabel('Feature 1')
# ax1.set_xlabel('Feature 2')
# ax1.set_xticks(())
# ax1.set_yticks(())
# ax1.set_title(title)
# ax1.legend()
# plt.show()