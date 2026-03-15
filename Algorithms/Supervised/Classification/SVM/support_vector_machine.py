# =============================================================================
# Titanic Survival Prediction — Support Vector Machine (SVM)
# =============================================================================
# Goal   : Predict whether a passenger survived (1) or died (0)
# Dataset: Built into seaborn — no CSV needed
# Note   : Reusing the same Titanic dataset and cleaning pipeline from
#          the Logistic Regression and KNN examples
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing Support Vector Machine model libraries
from sklearn.svm import SVC
# Evaluation tools — same as Logistic Regression and KNN
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


warnings.filterwarnings('ignore')

df = sns.load_dataset('titanic')
# print(data_set)
#      survived  pclass     sex   age  ...  deck  embark_town  alive  alone
# 0           0       3    male  22.0  ...   NaN  Southampton     no  False
# 1           1       1  female  38.0  ...     C    Cherbourg    yes  False
# 2           1       3  female  26.0  ...   NaN  Southampton    yes   True
# 3           1       1  female  35.0  ...     C  Southampton    yes  False
# 4           0       3    male  35.0  ...   NaN  Southampton     no   True
# ..        ...     ...     ...   ...  ...   ...          ...    ...    ...
# 886         0       2    male  27.0  ...   NaN  Southampton     no   True
# 887         1       1  female  19.0  ...     B  Southampton    yes   True
# 888         0       3  female   NaN  ...   NaN  Southampton     no  False
# 889         1       1    male  26.0  ...     C    Cherbourg    yes   True
# 890         0       3    male  32.0  ...   NaN   Queenstown     no   True
#
# [891 rows x 15 columns]
#
# print(df.columns)
# ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
#        'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
#        'alive', 'alone']

# print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 15 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   survived     891 non-null    int64
#  1   pclass       891 non-null    int64
#  2   sex          891 non-null    object
#  3   age          714 non-null    float64
#  4   sibsp        891 non-null    int64
#  5   parch        891 non-null    int64
#  6   fare         891 non-null    float64
#  7   embarked     889 non-null    object
#  8   class        891 non-null    category
#  9   who          891 non-null    object
#  10  adult_male   891 non-null    bool
#  11  deck         203 non-null    category
#  12  embark_town  889 non-null    object
#  13  alive        891 non-null    object
#  14  alone        891 non-null    bool
# dtypes: bool(2), category(2), float64(2), int64(4), object(5)

# looks like there are null values let me confirm
# print(df.isnull().sum())
# survived         0
# pclass           0
# sex              0
# age            177
# sibsp            0
# parch            0
# fare             0
# embarked         2
# class            0
# who              0
# adult_male       0
# deck           688
# embark_town      2
# alive            0
# alone            0
# dtype: int64

# -----------------------------------------------------------------------------
# Drop 6 columns before doing anything
# -----------------------------------------------------------------------------
# deck        — 688 out of 891 values blank; filling with mean = making numbers up
# alive       — just the survived column written as yes/no; keeping it = cheating
# class       — same thing as pclass, just written differently; duplicate
# embark_town — same thing as embarked, just written differently; duplicate
# who         — already captured by sex + age combined; adds nothing new
# adult_male  — same; already captured by sex + age; redundant
# -----------------------------------------------------------------------------

df.drop(['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], axis=1, inplace=True)

# print(df.info())
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 9 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   survived  891 non-null    int64
#  1   pclass    891 non-null    int64
#  2   sex       891 non-null    object
#  3   age       714 non-null    float64
#  4   sibsp     891 non-null    int64
#  5   parch     891 non-null    int64
#  6   fare      891 non-null    float64
#  7   embarked  889 non-null    object
#  8   alone     891 non-null    bool
# dtypes: bool(1), float64(2), int64(4), object(2)

df['age'].fillna(df['age'].mean(), inplace=True)  # 177 missing — filled with mean (~29.7); better than dropping 177 rows
# print(df['age'])
# 0      22.000000
# 1      38.000000
# 2      26.000000
# 3      35.000000
# 4      35.000000
#          ...
# 886    27.000000
# 887    19.000000
# 888    29.699118
# 889    26.000000
# 890    32.000000
# Name: age, Length: 891, dtype: float64

# embarked only has 2 missing rows — just drop them entirely
# losing 2 rows out of 891 is negligible: 891 → 889
df.dropna(subset=['embarked'], inplace=True)  # no axis= argument means rows are dropped by default

# print(df.info())
# Index: 889 entries, 0 to 890
# Data columns (total 9 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   survived  889 non-null    int64
#  1   pclass    889 non-null    int64
#  2   sex       889 non-null    object
#  3   age       889 non-null    float64
#  4   sibsp     889 non-null    int64
#  5   parch     889 non-null    int64
#  6   fare      889 non-null    float64
#  7   embarked  889 non-null    object
#  8   alone     889 non-null    bool
# dtypes: bool(1), float64(2), int64(4), object(2)

# -----------------------------------------------------------------------------
# Encode text columns — models can only work with numbers, not words
# female=0, male=1  |  Cherbourg=0, Queenstown=1, Southampton=2
# LabelEncoder assigns numbers alphabetically
# -----------------------------------------------------------------------------
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])
#    survived  pclass  sex   age  sibsp  parch     fare  embarked  alone
# 0         0       3    1  22.0      1      0   7.2500         2  False
# 1         1       1    0  38.0      1      0  71.2833         0  False
# 2         1       3    0  26.0      0      0   7.9250         2   True
# 3         1       1    0  35.0      1      0  53.1000         2  False
# 4         0       3    1  35.0      0      0   8.0500         2   True

# alone is still True/False (bool) — convert the whole dataframe to int for uniformity
# True → 1, False → 0
df = df.astype(int)
# print(df.head())
#    survived  pclass  sex  age  sibsp  parch  fare  embarked  alone
# 0         0       3    1   22      1      0     7         2      0
# 1         1       1    0   38      1      0    71         0      0
# 2         1       3    0   26      0      0     7         2      1
# 3         1       1    0   35      1      0    53         2      0
# 4         0       3    1   35      0      0     8         2      1

# -----------------------------------------------------------------------------
# Define features and target
# X = everything the model is allowed to look at
# y = survived — the answer we're trying to predict
# -----------------------------------------------------------------------------
X = df.drop('survived', axis=True)
y = df['survived']

# 80% of rows go to training, 20% held back for the final evaluation
# random_state=42 makes the split reproducible — same split every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# -----------------------------------------------------------------------------
# StandardScaler — required for SVM
# -----------------------------------------------------------------------------
# SVM finds the maximum-margin boundary by measuring distances between points.
# If fare ranges up to 500 and pclass only goes 1–3, fare would dominate the
# distance calculation and skew the boundary. Scaling brings all features to
# the same range so each one contributes equally.
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
model = SVC(kernel="rbf")  # model created
# SVM model also requires feature scale
model.fit(X_trained_scaled,y_train)
y_prediction = model.predict(X_test_scaled)
# print(y_prediction)
# [0 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1
#  1 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 0 1 0 1 0 0 0 1 1 0 1 1 0
#  0 1 0 0 1 0 1 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0
#  0 0 1 0 0 0 0 1 0 1 1 0 0 1 1 0 1 1 0 1 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 0 1
#  0 1 0 1 0 1 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 1 1 1 1 0 1 0 0 1]

accuracy_score = accuracy_score(y_test,y_prediction)
# print(accuracy_score)
# 0.8258426966292135
confusion_matrix = confusion_matrix(y_test,y_prediction)
# print(confusion_matrix)
# [[96 13]
#  [18 51]]
classification_report = classification_report(y_test,y_prediction)
# print(classification_report)
#               precision    recall  f1-score   support
#
#            0       0.84      0.88      0.86       109
#            1       0.80      0.74      0.77        69
#
#     accuracy                           0.83       178
#    macro avg       0.82      0.81      0.81       178
# weighted avg       0.82      0.83      0.82       178





