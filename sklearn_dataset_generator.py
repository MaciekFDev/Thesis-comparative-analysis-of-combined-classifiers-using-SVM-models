from matplotlib import pyplot as plt
import sklearn as sk
from sklearn import datasets, preprocessing
import numpy as np
import matplotlib.patches as mpatches
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import os
import csv
import pandas as pd

#####################
# Software in charge of generating data with use of Scikit-learn library
###

# Datasets' generator

# Standard blobs sets
X_class, y_class = sk.datasets.make_classification(n_samples = 5000)

# Moon sets
X_moon, y_moon = sk.datasets.make_moons(n_samples = 5000, noise = 0.1)

# Circle sets
X_circle, y_circle = sk.datasets.make_circles(n_samples = 5000, noise = 0.05)

# Classification dataset
X_gauss, y_gauss = sk.datasets.make_gaussian_quantiles(n_samples = 5000, n_classes = 2)

# Plotting all datasets
plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(411)
plt.title("Classification dataset")
plt.scatter(X_class[:,0], X_class[:,1], marker='o', c=y_class, s=25, edgecolor='k')

plt.subplot(412)
plt.title("Moon dataset")
plt.scatter(X_moon[:,0], X_moon[:,1], marker='o', c=y_moon, s=25, edgecolor='k')

plt.subplot(413)
plt.title("Circles dataset")
plt.scatter(X_circle[:,0], X_circle[:,1], marker='o', c=y_circle, s=25, edgecolor='k')

plt.subplot(414)
plt.title("Gaussian quantiles dataset")
plt.scatter(X_gauss[:,0], X_gauss[:,1], marker='o', c=y_gauss, s=25, edgecolor='k')

plt.show()

np.savetxt('X_class_file5.txt', X_class, delimiter = ',')
np.savetxt('y_class_file5.txt', y_class, delimiter = ',')

np.savetxt('X_moon_file5.txt', X_moon, delimiter = ',')
np.savetxt('y_moon_file5.txt', y_moon, delimiter = ',')

np.savetxt('X_circle_file5.txt', X_circle, delimiter = ',')
np.savetxt('y_circle_file5.txt', y_circle, delimiter = ',')

np.savetxt('X_gauss_file5.txt', X_gauss, delimiter = ',')
np.savetxt('y_gauss_file5.txt', y_gauss, delimiter = ',')
