import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# we will be inbuilt data set known as iris, it is a non-binary dataset
# having a lot of dimensions

df = sns.load_dataset('iris')
# print(df.head())
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa

# we do not have to clean the dataset as it is already preprocessed, and we can use it directly.

# print(df['species'].unique())
# ['setosa' 'versicolor' 'virginica']

# splitting the data in x and y variable

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# not using cross validation here as of now, might use in latter part of code to compare


model_KNN = KNeighborsClassifier(n_neighbors=13)
model_KNN.fit(X_train, y_train)  # model created here
# instead of predicting with model we can directly check the score of model
KNN_score = model_KNN.score(X_test, y_test)
# print(KNN_score)
# 0.98 score with n_neighbors=5
# -------------------------------
# print(KNN_score)
# 1.0 score with n_neighbors=13, but this means the model is overfitting
# changing the hyperparameters affect the model performance

#  let's use SVM model now and use same approach and see the performance

model_SVM = SVC(gamma='auto', C=10, kernel='linear')
model_SVM.fit(X_train, y_train)
score_svm = model_SVM.score(X_test, y_test)
# print(score_svm) with cv =30 and kernel as rbf
# 0.98
# print(score_svm) with cv as 10 and kernel as linear
# 0.98
# --------------------------------
# i understood that changing hyperparameters is causing change
# in accuracy score of model, but changing it manually is tedious
# task and I don't know which will be best so how to choose which
# will be best, for this I will use Randomised Search CV
