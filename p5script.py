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

# initial reading in of data & such
data_raw = pd.DataFrame(pd.read_csv('train.csv'))
len(data_raw)

data_raw.sum()
data_raw.head()
data_raw.dtypes

data_raw.describe()

# histrogram of age, dropped NA age values
age_hist = data_raw['Age'].sort_values()
age_hist.dropna(inplace=True)
plt.hist(age_hist)

# plotting distribution of fare values with high value outliers removed
fare_hist = data_raw['Fare'].sort_values()
fare_hist = fare_hist.where(fare_hist < 300).sort_values()
fare_hist.dropna(inplace=True)
sns.distplot(fare_hist, kde= False, bins = 100)

# dummy variables for sex & clean the unecessary columns from the data (also 1 other categorical)
sex_dummies = pd.get_dummies(data_raw['Sex'])
data_raw= data_raw.drop(['Sex'], axis=1)
data_raw = pd.concat([data_raw, sex_dummies], axis=1)
data_raw

data_raw = data_raw.drop(['Cabin', 'Ticket', 'Name'], axis=1)
class_dummies = pd.get_dummies(data_raw['Pclass'])
data_raw = data_raw.drop(['Pclass'], axis=1)
data_raw = pd.concat([data_raw, class_dummies], axis=1)

data_raw.dropna(inplace=True)
# separating data
mask = ['Age', 'SibSp', 'Parch', 'Fare', 'female', 'male', 1, 2, 3]

y = data_raw['Survived']
X = data_raw[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# fitting logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
X.columns
lr.coef_

# logistic regression with regularized numerical variables

# scaling data
mask_num = ['Age', 'SibSp', 'Parch', 'Fare']
mask_cat = ['female', 'male', 1, 2, 3]
data_scaled = data_raw[mask_num]
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_scaled))
data_scaled.columns = mask_num
data_scaled = pd.concat([data_scaled, data_raw[mask_cat]], axis =1)

# scaled numerical training & testing data
X2 = data_scaled
y2 = data_raw['Survived']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2)
lr2 = LogisticRegression()
lr2.fit(X2_train, y2_train)
lr2.score(X2_test, y2_test)

# scaling of data decreases my score by ~15% lolz

# predicting labels based on test set (using generic LogReg model with non-scaled data)
logreg_preds = lr.predict(X_test)
# predicting probability of each set of test variables
lr.predict_proba(X_test)

# cross validating model
cross_val_score(lr, X, y, cv=5)
# classification_report
print classification_report(y_test, logreg_preds)

# confusion matrix
print confusion_matrix(y_test, logreg_preds)

# plotting ROC curve for logreg

proba = lr.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, proba[:,1])

plt.figure()
plt.plot(fpr, tpr, label= 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of LogReg Model')
plt.show()


# gridsearch CV for optimal logreg paramaters

logreg_parameters = {
    'penalty':['l1','l2'],
    'C':np.logspace(-5,1,50),
    'solver':['liblinear']}

classifier = LogisticRegression()
gridsearch = GridSearchCV(classifier, logreg_parameters, cv=5)
gridsearch.fit(X, y)
gridsearch.best_score_
gridsearch.best_params_
gridsearch.param_grid

# KNN classifier
kvals = range(1,51)
neigh = KNeighborsClassifier()
neigh_parameters = {'n_neighbors':kvals, 'weights':['uniform','distance']}
neighbors_gridsearch = GridSearchCV(neigh, neigh_parameters, cv=5)
neighbors_gridsearch.fit(X, y)

neighbors_gridsearch.best_params_
