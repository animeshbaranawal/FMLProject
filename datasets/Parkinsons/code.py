from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,f1_score,roc_curve
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas
import pickle
import numpy as np

print(__doc__)

dataframe = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data')

Y = dataframe[['status']]
dataframe = dataframe.drop(['status'], axis = 1)
dataframe = dataframe.drop(['name'], axis = 1)
df_norm = (dataframe - dataframe.mean()) / (dataframe.max() - dataframe.min())

Y = np.ravel(Y)
X = dataframe.values

test_size = 0.3
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# print(y_train)

print('--------------------- SVM ---------------------------------')
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
					 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']

for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()

	clf = GridSearchCV(SVC(random_state = 0), tuned_parameters, scoring='%s_macro' % score)
	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print("Detailed classification report:")
	print()
	print()
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))
	print('train accuracy: '+str(clf.score(X_train,y_train)))
	print('test accuracy: '+str(clf.score(X_test,y_test)))
	print()

print('\n\n\n\n-------------------------- Decision Trees ------------------')

tuned_parameters = [{'criterion': ['gini','entropy'], 'min_samples_split' : [2,5,10,15,20], 'min_samples_leaf' : [1,2,5,10,20], 'max_features' : [4,5,8,10,12,14,16,18,20,22]}]

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Detailed classification report:")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print('train accuracy: '+str(clf.score(X_train,y_train)))
print('test accuracy: '+str(clf.score(X_test,y_test)))   
print()


print('\n\n\n\n-------------------------- Random Forests ------------------')
tuned_parameters = [{'n_estimators': [4,8,15,20,50,100], 'max_depth':[2,5,7,9,10]}]
scores = ['precision', 'recall']

for score in scores:
	print("# Tuning hyper-parameters for %s" % score)
	print()

	clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)
	clf.fit(X_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print("Detailed classification report:")
	print()
	y_true, y_pred = y_test, clf.predict(X_test)
	print(classification_report(y_true, y_pred))
	print('train accuracy: '+str(clf.score(X_train,y_train)))
	print('test accuracy: '+str(clf.score(X_test,y_test)))   
	print()