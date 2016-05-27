import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

%matplotlib inline

data_raw = pd.DataFrame(pd.read_csv('train.csv'))

sex_dummies = pd.get_dummies(data_raw['Sex'])
data_raw= data_raw.drop(['Sex'], axis=1)
data_raw = pd.concat([data_raw, sex_dummies], axis=1)


data_raw = data_raw.drop(['Cabin', 'Ticket', 'Name'], axis=1)
class_dummies = pd.get_dummies(data_raw['Pclass'])
data_raw = data_raw.drop(['Pclass'], axis=1)
data_raw = pd.concat([data_raw, class_dummies], axis=1)

mean = data_raw['Age'].mean()
data_raw['Age']= data_raw['Age'].fillna(mean)
data_raw.isnull().sum()
data_raw.dropna(inplace=True)

mask = ['Age', 'SibSp', 'Parch', 'Fare', 'female', 'male', 1, 2, 3]
y = data_raw['Survived']
X = data_scaled

kvals = range(1,51)
neigh = KNeighborsClassifier()
neigh_parameters = {'n_neighbors':kvals, 'weights':['uniform','distance']}
neighbors_gridsearch = GridSearchCV(neigh, neigh_parameters, cv=5)
neighbors_gridsearch.fit(X, y)

neighbors_gridsearch.best_score_

mask_num = ['Age', 'SibSp', 'Parch', 'Fare']
mask_cat = ['female', 'male', 1, 2, 3]
data_scaled = data_raw[mask_num]
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled))
data_scaled.columns = mask_num
data_scaled = pd.concat([data_scaled, data_raw[mask_cat]], axis =1)

data_scaled.dropna(inplace=True)
len(data_scaled)
len(y)
