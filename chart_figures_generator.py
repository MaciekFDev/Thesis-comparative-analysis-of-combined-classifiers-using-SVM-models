from matplotlib import pyplot as plt
import sklearn as sk
from sklearn import datasets, preprocessing
import numpy as np
import matplotlib.patches as mpatches


# Software will be used to generate exemplary charts which will be used in thesis' figures.
# Creating sets for exemplary points (objects of classification)


# For one-dimensional sets
tab = np.arange(18)
tab0 = np.zeros(18)

# For two-dimensional sets
X, y = sk.datasets.make_blobs(n_samples = 18, centers = 2)

# Plotting part
plt.scatter(tab, tab0, marker='o', c=np.array(['yellow', 'red'])[y], s=25, edgecolor='k', label='Class 1 and Class 2')
pop_a = mpatches.Patch(color='yellow', label="Class 1")
pop_b = mpatches.Patch(color='red', label="Class 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(handles=[pop_a, pop_b])

plt.show()