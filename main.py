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

#####################
# Software in charge of generating data with use of Scikit-learn library
###

# Datasets' generator

# Standard blobs sets
X_blobs, y_blobs = sk.datasets.make_blobs(n_samples = 5000, centers = 2)

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
plt.title("Blobs dataset")
plt.scatter(X_blobs[:,0], X_blobs[:,1], marker='o', c=y_blobs, s=25, edgecolor='k')

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

# Saving generated data to file for further processing
# Saving blobs
with open('X_blobs_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerows(X_blobs)

with open('y_blobs_file.csv', mode='w') as file:
    csv.writer(file, delimiter = ' ').writerow(y_blobs)

# Saving moons
with open('X_moon_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerows(X_moon)

with open('y_moon_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerow(y_moon)

# Saving circles
with open('X_circle_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerows(X_circle)

with open('y_circle_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerow(y_circle)

# Saving gaussian quantiles
with open('X_gauss_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerows(X_gauss)

with open('y_gauss_file.csv', 'w') as file:
    csv.writer(file, delimiter = ' ').writerow(y_gauss)
