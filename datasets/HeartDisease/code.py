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

names = ['age','sex','chestpaintype','resting_blood_pressure','serum_cholestrol','fasting_blood_sugar','resting_ecg','max_heart_rate','exercise_induced_angina','st_depression_induced_by_exercise','slope_of_peak_exercise','number_of_major_vessel','thal','heart_disease_diag']
dataframe = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', names=names)

# replace missing values
dataframe = dataframe.replace({'?': 'NaN'}, regex=False)
imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(dataframe[['number_of_major_vessel']].values)
dataframe[['number_of_major_vessel']] = imp.transform(dataframe[['number_of_major_vessel']].values)
imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(dataframe[['thal']].values)
dataframe[['thal']] = imp.transform(dataframe[['thal']].values)


Y = dataframe[['heart_disease_diag']]
dataframe = dataframe.drop(['heart_disease_diag'], axis = 1)
Y['heart_disease_diag'][Y['heart_disease_diag'] > 0] = 1
x = feature_selection.mutual_info_classif(dataframe,np.ravel(Y))
print(x)

Y = np.ravel(Y)

# one hot encode sex column
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['sex']].values)
x = enc.transform(dataframe[['sex']].values).toarray()

dataframe = dataframe.drop(['sex'], axis = 1)
dataframe['sex_0'] = x[:,0]
dataframe['sex_1'] = x[:,1]

# one hot encode chest pain
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['chestpaintype']].values)
x = enc.transform(dataframe[['chestpaintype']].values).toarray()
dataframe = dataframe.drop(['chestpaintype'], axis = 1)
dataframe['cpt_0'] = x[:,0]
dataframe['cpt_1'] = x[:,1]
dataframe['cpt_2'] = x[:,2]
dataframe['cpt_3'] = x[:,3]

# one hot encode exercise_induced_angina
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['exercise_induced_angina']].values)
x = enc.transform(dataframe[['exercise_induced_angina']].values).toarray()
dataframe = dataframe.drop(['exercise_induced_angina'], axis = 1)
dataframe['eia_0'] = x[:,0]
dataframe['eia_1'] = x[:,1]

# one hot encode number_of_major_vessel
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['number_of_major_vessel']].values)
x = enc.transform(dataframe[['number_of_major_vessel']].values).toarray()
dataframe = dataframe.drop(['number_of_major_vessel'], axis = 1)
dataframe['nmv_0'] = x[:,0]
dataframe['nmv_1'] = x[:,1]
dataframe['nmv_2'] = x[:,2]
dataframe['nmv_3'] = x[:,3]

# one hot encode slope_of_peak_exercise
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['slope_of_peak_exercise']].values)
x = enc.transform(dataframe[['slope_of_peak_exercise']].values).toarray()
dataframe = dataframe.drop(['slope_of_peak_exercise'], axis = 1)
dataframe['spe_0'] = x[:,0]
dataframe['spe_1'] = x[:,1]
dataframe['spe_2'] = x[:,2]

# one hot encode fasting_blood_sugar
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['fasting_blood_sugar']].values)
x = enc.transform(dataframe[['fasting_blood_sugar']].values).toarray()
dataframe = dataframe.drop(['fasting_blood_sugar'], axis = 1)
dataframe['fbs_0'] = x[:,0]
dataframe['fbs_1'] = x[:,1]

# one hot encode resting_ecg
enc = preprocessing.OneHotEncoder()
enc.fit(dataframe[['resting_ecg']].values)
x = enc.transform(dataframe[['resting_ecg']].values).toarray()
dataframe = dataframe.drop(['resting_ecg'], axis = 1)
dataframe['recg_0'] = x[:,0]
dataframe['recg_1'] = x[:,1]
dataframe['recg_2'] = x[:,2]

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

tuned_parameters = [{'criterion': ['gini','entropy'], 'min_samples_split' : [2,5,10,15,20], 'min_samples_leaf' : [1,2,5,10,20], 'max_features' : [4,5,8,10,12,14,16,18,20,26]}]

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