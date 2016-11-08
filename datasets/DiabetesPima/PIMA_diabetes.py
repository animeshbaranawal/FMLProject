import pandas
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
# model = LogisticRegression(C=4.8942898961145316)
model = RandomForestClassifier(n_estimators= 20, random_state=0, max_depth= 7)
model.fit(X_train, Y_train)
# save the model to disk
filename = 'Random_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# Set the parameters by cross-validation
# tuned_parameters = [{'n_estimators': [4,8,15,20], 'max_depth':[2,5,7,9,10]}]
# tuned_parameters = [{'C':np.logspace(-5,2,1000)}]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)
#     clf.fit(X_train, Y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     #     print("%0.3f (+/-%0.03f) for %r"
#     #           % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = Y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()



# # load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)