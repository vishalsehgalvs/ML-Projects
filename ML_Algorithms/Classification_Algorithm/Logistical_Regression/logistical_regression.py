# titanic passenger data — predicting who survived and who didn't
# dataset is built into seaborn, no separate CSV needed

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# to measure how well the model did after training
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

# dropping 6 columns before doing anything:
# deck       — 688 out of 891 blank, filling with mean would just be making numbers up
# alive      — this is just the survived column written as yes/no, keeping it would be cheating
# class      — same thing as pclass, duplicate
# embark_town — same thing as embarked, duplicate
# who, adult_male — already captured by sex and age, nothing new here

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

df['age'].fillna(df['age'].mean(), inplace=True)  # 177 ages missing — filled with mean (~29.7)
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

# embarked only has 2 missing rows — just dropping them
# losing 2 rows out of 891 is nothing, 891 → 889

df.dropna(subset=['embarked'], inplace=True)  # no axis means rows get dropped by default

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

# sex and embarked are still words — model can't use words, turning them into numbers
# female=0, male=1  |  Cherbourg=0, Queenstown=1, Southampton=2
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])
#    survived  pclass  sex   age  sibsp  parch     fare  embarked  alone
# 0         0       3    1  22.0      1      0   7.2500         2  False
# 1         1       1    0  38.0      1      0  71.2833         0  False
# 2         1       3    0  26.0      0      0   7.9250         2   True
# 3         1       1    0  35.0      1      0  53.1000         2  False
# 4         0       3    1  35.0      0      0   8.0500         2   True

# alone column is still True/False — converting everything to int to keep it uniform

df = df.astype(int)
# print(df.head())
#    survived  pclass  sex  age  sibsp  parch  fare  embarked  alone
# 0         0       3    1   22      1      0     7         2      0
# 1         1       1    0   38      1      0    71         0      0
# 2         1       3    0   26      0      0     7         2      1
# 3         1       1    0   35      1      0    53         2      0
# 4         0       3    1   35      0      0     8         2      1

# X = all input columns, y = survived (the thing we're trying to guess)

X = df.drop('survived', axis=True)
y = df['survived']

# 80% rows go into training, 20% held back for the final test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)  # feeds the training rows in, model figures out the pattern

y_predict = model.predict(X_test)
# print(y_predict)
# [0 1 1 0 1 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 1 0 0 1
#  1 0 0 0 1 0 0 0 0 1 0 0 1 1 1 0 0 1 1 1 0 0 0 0 1 1 0 1 0 0 0 1 1 0 1 1 0
#  1 1 0 0 1 0 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0
#  1 0 1 0 0 0 0 1 0 0 1 0 0 1 1 1 1 1 0 1 0 0 0 1 1 0 1 1 1 1 0 1 0 1 0 0 1
#  0 1 0 1 0 1 0 1 0 0 0 1 0 1 0 0 0 1 1 0 0 1 1 1 1 0 1 0 0 1]

# print(y_test)
# 281    0
# 435    1
# 39     1
# 418    0
# 585    1
#       ..
# 433    0
# 807    0
# 25     1
# 85     1
# 10     1
# Name: survived, Length: 178, dtype: int64

# linear regression used R² to measure accuracy — that only works when predicting a number
# here the answer is 0 or 1, so we check how many we got right vs wrong
# the tool for that is a confusion matrix

# confusion matrix breaks it into 4 buckets:
# said 0, was 0 → correct  (True Negative)
# said 1, was 1 → correct  (True Positive)
# said 1, was 0 → wrong    (False Positive — false alarm)
# said 0, was 1 → wrong    (False Negative — missed a survivor)
# from those 4 numbers we get: Accuracy, Precision, Recall, F1

# =============================================
# model evaluation
# =============================================

accuracy_score = accuracy_score(y_test, y_predict)
# print(accuracy_score)
# 0.8033707865168539
cm = confusion_matrix(y_test, y_predict)
# print(cm)
# [[90 19]
#  [16 53]]

cr = classification_report(y_test, y_predict)
# print(cr)
#               precision    recall  f1-score   support
#
#            0       0.85      0.83      0.84       109
#            1       0.74      0.77      0.75        69
#
#     accuracy                           0.80       178
#    macro avg       0.79      0.80      0.79       178
# weighted avg       0.81      0.80      0.80       178


