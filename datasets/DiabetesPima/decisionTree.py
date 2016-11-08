
from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,f1_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
import pandas
import pickle

print(__doc__)

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv('pima-indians-diabetes.data', names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Set the parameters by cross-validation
tuned_parameters = [{'criterion': ['gini','entropy'], 'min_samples_split' : [2,5,10,15,20], 'min_samples_leaf' : [1,2,5,10,20], 'max_features' : [4,5,6,7,8]}]

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.

#Best features found - {'max_features': 6, 'min_samples_split': 10, 'criterion': 'gini', 'min_samples_leaf': 10}
best_model = DecisionTreeClassifier(max_features=6,min_samples_leaf=10,min_samples_split=10,criterion='gini')
best_model.fit(X_train, y_train)

print('----scoring on train----')
accuracy = best_model.score(X_train, y_train)
mf1_score = f1_score(y_train,best_model.predict(X_train))
mroc = roc_curve(y_train,best_model.predict(X_train))
print('accuracy: ' + str(accuracy)+'\nf1_score: '+str(mf1_score)+'\nroc: '+str(mroc))


print('----scoring on test----')
accuracy = best_model.score(X_test, y_test)
mf1_score = f1_score(y_test,best_model.predict(X_test))
mroc = roc_curve(y_test,best_model.predict(X_test))
print('accuracy: ' + str(accuracy)+'\nf1_score: '+str(mf1_score)+'\nroc: '+str(mroc))

filename = 'decisionTree.sav'
pickle.dump(best_model, open(filename, 'wb'))