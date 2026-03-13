# =============================================================================
# Titanic Survival Prediction — Decision Tree
# =============================================================================
# Goal   : Predict whether a passenger survived (1) or died (0)
# Dataset: Built into seaborn — no CSV needed
# Note   : Same Titanic dataset and cleaning pipeline as previous algorithms.
#          Referring same titanic data set for model tuning and demonstrating
#           K- fold cross validation
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
# importing state vector machine model libraries
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# DecisionTreeClassifier — builds a tree of if/else questions to classify passengers
from sklearn.tree import DecisionTreeClassifier

# Evaluation tools — same as previous algorithms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# StandardScaler — imported and used before fitting
from sklearn.preprocessing import StandardScaler

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
# deck        — 688 out of 891 blank; too much missing data to be useful
# alive       — same as survived written as yes/no; keeping it = cheating
# class       — same as pclass, different label; duplicate
# embark_town — same as embarked, different label; duplicate
# who         — just restates what sex + age already tell us
# adult_male  — same; already captured by sex + age
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

df['age'].fillna(df['age'].mean(), inplace=True)  # 177 missing — filled with mean (~29.7)
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

# embarked only has 2 missing rows — drop them entirely
# going from 891 to 889 rows is not a meaningful loss
df.dropna(subset=['embarked'], inplace=True)  # no axis= argument — rows are dropped by default

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
# Encode text columns — models only understand numbers, not words
# female=0, male=1  |  Cherbourg=0, Queenstown=1, Southampton=2
# LabelEncoder assigns integers alphabetically
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

# alone is stored as True/False — convert the whole dataframe to int for uniformity
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
# X = the 8 columns the model is allowed to look at
# y = survived — the answer we're predicting
# -----------------------------------------------------------------------------
X = df.drop('survived', axis=True)
y = df['survived']

model_svc = SVC()
model_KNN = KNeighborsClassifier()

# -------------------------------------------------------
# Default train test split was used 80% and 20% last time
# now i will use Cross Validation - K fold cross validation
# from sklearn.model_selection import cross_val_score
# -------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# scores = cross_val_score(model_svc, X_scaled, y, cv=5, scoring='accuracy')
# print(scores)
# [0.83146067 0.82022472 0.81460674 0.80898876 0.86440678] -> multiple accuracy

# scores = scores.mean()
# print(scores)
# 0.8279375357074844

scores = cross_val_score(model_KNN, X_scaled, y, cv=5, scoring='accuracy')
# print(scores)
# print(scores.mean())
# [0.78651685 0.76404494 0.8258427  0.81460674 0.80225989]
# 0.7986542245921411
# -----------------------------------------------------------------------------
# Model evaluation
# -----------------------------------------------------------------------------
# accuracy_score  — overall % of correct predictions'
# confusion_matrix — breaks predictions into 4 buckets:
#   [TN  FP]   TN = predicted died, was dead     (correct)
#   [FN  TP]   TP = predicted survived, did       (correct)
#              FP = predicted survived, actually died  (false alarm)
#              FN = predicted died, actually survived  (missed)
# classification_report — precision, recall, f1 per class
# -----------------------------------------------------------------------------

# accuracy_score = accuracy_score(y_test, y_predictions)
# print(accuracy_score)
# 0.7696629213483146  →  77.0% accuracy

# confusion_matrix = confusion_matrix(y_test, y_predictions)
# print(confusion_matrix)
# [[88  21]   — 88 correct deaths, 21 false alarms
#  [20  49]]  — 20 missed survivors, 49 correct survivors

# classification_report = classification_report(y_test, y_predictions)
# print(classification_report)
#               precision    recall  f1-score   support
#
#            0       0.81      0.81      0.81       109
#            1       0.70      0.71      0.71        69
#
#     accuracy                           0.77       178
#    macro avg       0.76      0.76      0.76       178
# weighted avg       0.77      0.77      0.77       178

# -------------------------------------------------------
# Default train test split was used 80% ans 20%
# -------------------------------------------------------
