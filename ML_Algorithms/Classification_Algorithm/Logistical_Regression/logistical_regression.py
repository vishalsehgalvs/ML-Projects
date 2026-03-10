#  inbuild titanic data set is used from seaborn library

# importing basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ==============model evaluation libraries===========
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

# deck has lot of nul values no sense in using mean to fill the value sas it will result in bad model

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

df['age'].fillna(df['age'].mean(), inplace=True)  # filling na values with mean of age
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

# for 2 missing values in embark we can either fill them with mean value of column or drop entire missing entries
# so now total entries will come down to 889

df.dropna(subset=['embarked'], inplace=True)  # axis not selected so by default axis will be 0

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

# now we need to perform label encoding of non-numeric data ->from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])
#    survived  pclass  sex   age  sibsp  parch     fare  embarked  alone
# 0         0       3    1  22.0      1      0   7.2500         2  False
# 1         1       1    0  38.0      1      0  71.2833         0  False
# 2         1       3    0  26.0      0      0   7.9250         2   True
# 3         1       1    0  35.0      1      0  53.1000         2  False
# 4         0       3    1  35.0      0      0   8.0500         2   True

# as you see the values are in true/false so we need to convert them to int using astype()

df = df.astype(int)
# print(df.head())
#    survived  pclass  sex  age  sibsp  parch  fare  embarked  alone
# 0         0       3    1   22      1      0     7         2      0
# 1         1       1    0   38      1      0    71         0      0
# 2         1       3    0   26      0      0     7         2      1
# 3         1       1    0   35      1      0    53         2      0
# 4         0       3    1   35      0      0     8         2      1

# split the data into x and y columns now to train and predict

X = df.drop('survived', axis=True)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)  # logistic regression model is ready

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

#  in linear regression we used to create r2 and adjusted_r2 to evaluate our
#   model but here we will create confusion matrix as its a classification problem

# confusion matrix gives us true positive , false positive, false negative and false positive quadrants
# from confusion matrix we calculate
# 1.Accuracy
# 2.Precision
# 3.Recall
# 4.F1 score

# =========Model evaluation===========

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


