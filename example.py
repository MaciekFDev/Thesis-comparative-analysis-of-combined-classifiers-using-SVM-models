from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#Task 1

data = np.genfromtxt("stream.csv", delimiter=",")
X = data[:,:-1]
y = data[:,-1]

#Cross validation part
kv = StratifiedKFold(n_splits=10)
scores = np.zeros((10))
for fold_id, (train_index, test_index) in enumerate (kv.split(X, y)):
	#print("TRAIN: ", train_index, "TEST: ", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	clf = SVC()
	clf.fit(X_train, y_train)
	scores[fold_id] = accuracy_score(y_test, clf.predict(X_test))

print(scores)
datamean = np.mean(scores)
datastd = np.std(scores)
print("Mean classification value and standard deviation for cross validation: %.3f" % datamean,"%.2f" % datastd)

#Part with ensemblies
scores2 = np.zeros((4))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#AdaBoost
AdaB = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='linear'), n_estimators=10)
AdaB.fit(X_train, y_train)
scores2[0] = accuracy_score(y_test, AdaB.predict(X_test))

#Bagging
BilboBclf = BaggingClassifier(base_estimator=SVC(), n_estimators=10,random_state=0)
BilboBclf.fit(X_train, y_train)
scores2[1] = accuracy_score(y_test, BilboBclf.predict(X_test))

#Voting
clf1 = SVC(probability=True)
clf2 = GaussianNB()
clf3 = KNeighborsClassifier(n_neighbors=3)

Vclf1 = VotingClassifier(estimators=[
        ('sv', clf1), ('gnb', clf2), ('knn', clf3)], voting='hard')
Vclf1.fit(X_train, y_train)
scores2[2] = accuracy_score(y_test, Vclf1.predict(X_test))

Vclf2 = VotingClassifier(estimators=[
        ('sv', clf1), ('gnb', clf2), ('knn', clf3)], voting='soft')
Vclf2.fit(X_train, y_train)
scores2[3] = accuracy_score(y_test, Vclf2.predict(X_test))

print(scores2)
datamean = np.mean(scores2)
datastd = np.std(scores2)
print("Mean classification value and standard deviation for ensemblies: %.3f" % datamean,"%.2f" % datastd)

#From the performed experiments we can see that best scores can be achieved for voting
#classifier in soft mode. Other ensemblies may achieve similar accuracy, regarding AdaBoostClassifier
#which could only achieve around 50% of accuracy. Accuracy score from voting classifier has almost
#  the same values as mean value from cross validation.