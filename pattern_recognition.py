from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import time

training_images = []                # half of pictures of each object
testing_images = []                 # second half, remaining pictures
edges_training = []                 # analogic variables, but for edges for particular images
edges_testing = []
Hu_training = []                    # analogic variables, but for hu's invariant moments
Hu_testing = []
training_images_ids = []            # variables for splitting images into rightful sets
testing_images_ids = []
X_test_2 = []                       # temporary variables for testing stage
y_test_2 = []
score_single_classifier = []             # array for holding scores for each single classification for single SVM case
score_ensemble_adab_classifier = []      # array for holding scores for each single classification for AdaBoost SVM ensemble case
score_ensemble_bagging_classifier = []   # array for holding scores for each single classification for Bagging SVM ensemble case
datamean_adab = []                       # arrays for storing mean values of classification results
datamean_bagging = []
datamean_single = []
datamean = []
datastd_adab = []                        # arrays for storing standard deviation values of classification results
datastd_bagging = []
datastd_single = []
datastd = []
mod = 0                             # modifier, responsible for navigating between objects in feature extraction part
nr_array = np.arange(0, 356, 5)     # array with numbers for randomizing images in both sets
test_array = np.arange(0, 3601, 36) # array arranged for picking testing classes for svm
obj = 'obj'                         # variable responsible for navigation between objects' images in images loading stage
                                    # Path to folder containing objects' images at local repository
path = 'C:/Users/Veteran/Thesis-combined-svm-algorithms-analysis/Datasets/coil-100/'

# Feature extraction from images
for i in range(1,51,1):        # loop over particular objects' folders
    obj_nr = obj + str(i)       # storing object's number (label)

    # Creating two sets of earlier separated images
    #np.random.shuffle(nr_array) # randomizing images of particular object
    #training_images_ids = nr_array[:len(nr_array)//2] # Randomization technique
    #testing_images_ids = nr_array[len(nr_array)//2:] 

    for k in range(72):         # Selecting picture every 10 degrees for training, and rest for testing
        if k % 2 == 0:
            training_images_ids.append(nr_array[k])
        else:
            testing_images_ids.append(nr_array[k])

    for k in range(36):         # loop over particular images
        # Building paths according to earlier established sets
        objpath_train = path + obj_nr + '__' + str(training_images_ids[k]) + '.png'
        objpath_test = path + obj_nr + '__' + str(testing_images_ids[k]) + '.png'

        # Loading images into separate sets (honouring labels - numbers of objects)
        training_image = cv2.imread(objpath_train, 0)
        training_images.append([training_image, i])
        testing_image = cv2.imread(objpath_test, 0)
        testing_images.append([testing_image, i])

    # Calculating edges for each image
    for k in range(36):
        edges_training.append([cv2.Canny(training_images[k+mod][0],100,200), i])
        edges_testing.append([cv2.Canny(testing_images[k+mod][0],100,200), i])
        #edges_training.append([training_images[k+mod][0], i])
        #edges_testing.append([testing_images[k+mod][0], i])

    # Calculating Hu's invariant values for each image
    for k in range(36):
        temp = np.array(cv2.HuMoments(cv2.moments(edges_training[k+mod][0])).flatten())
        temp = np.append(temp, [i])
        Hu_training.append(temp)
        temp = np.array(cv2.HuMoments(cv2.moments(edges_testing[k+mod][0])).flatten())
        temp = np.append(temp, [i])
        Hu_testing.append(temp)
    
    mod+=36                   # Jumping to next object

# Classification part
Hu_training = np.array(Hu_training)
Hu_testing = np.array(Hu_testing)

X_train = Hu_training[:,:-1]  # Splitting training set into atributes and labels
y_train = Hu_training[:,-1]

X_test = Hu_testing[:,:-1]    # Splitting testing set into atributes and labels
y_test = Hu_testing[:,-1]

clf = SVC(probability=True, kernel='linear')        # Initializing classifier
clf.fit(X_train, y_train)                           # Training given classifier
time.sleep(3)

AdaB = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='linear'),
                            n_estimators=10)        # Initializing classificators ensemble by using AdaBoost
AdaB.fit(X_train, y_train)                          # Training given ensemble
time.sleep(3)

Bagging = BaggingClassifier(base_estimator=SVC(),
                n_estimators=10,random_state=0)     # Initializing classificators ensemble by using Bagging
Bagging.fit(X_train, y_train)                       # Training given ensemble

training_object = 1188
time.sleep(3)
for j in range(0, 50, 1):                          # Testing stage for bagging SVM ensemble case - every image
    X_test_2.clear()                                # is being comparised to every other from testing set
    y_test_2.clear()
    for k in range(training_object, training_object+36, 1):
        X_test_2.append(X_test[k])
        y_test_2.append(y_test[k])

    for l in range(test_array[j], test_array[j]+36, 1):
        X_test_2.append(X_test[l])
        y_test_2.append(y_test[l])

    # Calculating accuracy for each classification and saving it in array
    score = accuracy_score(y_test_2, clf.predict(X_test_2))         # For individual SVM classifer
    score_single_classifier.append(round(score*100, 2))

time.sleep(3)
for j in range(0, 50, 1):                          # Testing stage for bagging SVM ensemble case - every image
    X_test_2.clear()                                # is being comparised to every other from testing set
    y_test_2.clear()
    for k in range(training_object, training_object+36, 1):
        X_test_2.append(X_test[k])
        y_test_2.append(y_test[k])

    for l in range(test_array[j], test_array[j]+36, 1):
        X_test_2.append(X_test[l])
        y_test_2.append(y_test[l])

    # Calculating accuracy for each classification and saving it in array
    score = accuracy_score(y_test_2, AdaB.predict(X_test_2))        # For AdaBoost SVM ensemble
    score_ensemble_adab_classifier.append(round(score*100, 2))

time.sleep(3)
for j in range(1, 50, 1):                          # Testing stage for bagging SVM ensemble case - every image
    X_test_2.clear()                                # is being comparised to every other from testing set
    y_test_2.clear()
    for k in range(training_object, training_object+36, 1):
        X_test_2.append(X_test[k])
        y_test_2.append(y_test[k])

    for l in range(test_array[j], test_array[j]+36, 1):
        X_test_2.append(X_test[l])
        y_test_2.append(y_test[l])

    # Calculating accuracy for each classification and saving it in array
    score = accuracy_score(y_test_2, Bagging.predict(X_test_2))     # For Bagging SVM ensemble
    score_ensemble_bagging_classifier.append(round(score*100, 2))

#for k in range(72,108): # For displaying images
 #   cv2.imshow('', edges_training[k][0])
 #   cv2.waitKey()

#for i in range(20):
    # Calculating mean values for each classification task case
  #  datamean_adab.append(np.mean(score_ensemble_adab_classifier[i*100:(i+1)*100]))
  #  datamean_bagging.append(np.mean(score_ensemble_bagging_classifier[i*100:(i+1)*100]))
  #  datamean_single.append(np.mean(score_single_classifier[i*100:(i+1)*100]))

    # Calculating standard deviation values for each classification task
  #  datastd_adab.append(np.std(score_ensemble_adab_classifier[i*100:(i+1)*100]))
  #  datastd_bagging.append(np.std(score_ensemble_bagging_classifier[i*100:(i+1)*100]))
  #  datastd_single.append(np.std(score_single_classifier[i*100:(i+1)*100]))

datamean.append(np.mean(score_ensemble_adab_classifier))
datamean.append(np.mean(score_ensemble_bagging_classifier))
datamean.append(np.mean(score_single_classifier))

datastd.append(np.std(score_ensemble_adab_classifier))
datastd.append(np.std(score_ensemble_bagging_classifier))
datastd.append(np.std(score_single_classifier))

print('Classification accuracy for single SVM (in %): ', score_single_classifier, '\n')
print('With mean value and standard deviation value: %.2f' % datamean[2], '%.2f' % datastd[2], '\n')

print('Classification accuracy for SVM ensemble (AdaBoost) (in %): ', score_ensemble_adab_classifier, '\n')
print('With mean value and standard deviation value: %.2f' % datamean[0], '%.2f' % datastd[0], '\n')

print('Classification accuracy for SVM ensemble (Bagging) (in %): ', score_ensemble_bagging_classifier, '\n')
print('With mean value and standard deviation value: %.2f' % datamean[1], '%.2f' % datastd[1], '\n')