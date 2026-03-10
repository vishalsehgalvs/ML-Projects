# Ford second hand car price prediction using Linear Regression
# dataset from kaggle — ford used car listings
# https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction

# the main thing i wanted to test here was whether one-hot encoding
# or label encoding gives a better result — spoiler: one-hot wins by a lot

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

df = pd.read_csv('ford.csv')
# print(df.head())
#      model  year  price transmission  mileage fuelType  tax   mpg  engineSize
# 0   Fiesta  2017  12000    Automatic    15944   Petrol  150  57.7         1.0
# 1    Focus  2018  14000       Manual     9083   Petrol  150  57.7         1.0
# 2    Focus  2017  13000       Manual    12456   Petrol  150  57.7         1.0
# 3   Fiesta  2019  17500       Manual    10460   Petrol  145  40.3         1.5
# 4   Fiesta  2019  16500    Automatic     1482   Petrol  145  48.7         1.0

# print(df.shape)
# (17966, 9)  — nearly 18k rows, 9 columns, decent size

# print(df.info())
# Data columns (total 9 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   model         17966 non-null  object
#  1   year          17966 non-null  int64
#  2   price         17966 non-null  int64
#  3   transmission  17966 non-null  object
#  4   mileage       17966 non-null  int64
#  5   fuelType      17966 non-null  object
#  6   tax           17966 non-null  int64
#  7   mpg           17966 non-null  float64
#  8   engineSize    17966 non-null  float64

# print(df.describe())
#                year         price  ...           mpg    engineSize
# count  17966.000000  17966.000000  ...  17966.000000  17966.000000
# mean    2016.866470  12279.534844  ...     57.906980      1.350807
# std        2.050336   4741.343657  ...     10.125696      0.432367
# min     1996.000000    495.000000  ...     20.800000      0.000000
# 25%     2016.000000   8999.000000  ...     52.300000      1.000000
# 50%     2017.000000  11291.000000  ...     58.900000      1.200000
# 75%     2018.000000  15299.000000  ...     65.700000      1.500000
# max     2060.000000  54995.000000  ...    201.800000      5.000000

# print(df.isnull().sum())  — no missing values at all, clean dataset
# model           0
# year            0
# price           0
# transmission    0
# mileage         0
# fuelType        0
# tax             0
# mpg             0
# engineSize      0

# 25%     2016.000000   8999.000000  ...     52.300000      1.000000
# 50%     2017.000000  11291.000000  ...     58.900000      1.200000
# 75%     2018.000000  15299.000000  ...     65.700000      1.500000
# max     2060.000000  54995.000000  ...    201.800000      5.000000
#
# [8 rows x 6 columns]

# print(df.isnull().sum()) - there are no null values
# model           0
# year            0
# price           0
# transmission    0
# mileage         0
# fuelType        0
# tax             0
# mpg             0
# engineSize      0
# dtype: int64


# EDA — all commented out, uncomment whichever one you want to look at

# price distribution — most cars cluster around the 8k-15k range
# sns.histplot(x= df['price'],kde = True, bins=50)
# plt.show()

# heatmap — mileage and year have the most obvious correlation with price
# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()

# petrol vs diesel vs hybrid — diesel tends to hold value better
# sns.boxplot(data = df, x= 'fuelType', y = 'price')
# plt.xticks(rotation = 90)
# plt.show()

# mileage vs price — higher mileage = cheaper, pretty obvious but good to confirm
# sns.scatterplot(data = df, x= 'mileage', y = 'price')
# plt.show()

# engine size vs price — bigger engine = pricier
# sns.boxplot(data=df, x= 'engineSize', y ='price')
# plt.show()

# transmission vs price — automatics are a bit more expensive on average
# sns.boxplot(data=df, x = 'transmission', y ='price')
# plt.show()

# model vs price — big spread here, some models clearly sell for more
# sns.boxplot(data=df, x = 'model', y ='price')
# plt.xticks(rotation = 90)
# plt.show()


# ============================================================
# APPROACH 1: One-Hot Encoding
# text columns like model, transmission, fuelType cant go into
# a model as words — need to turn them into numbers first.
# one-hot encoding creates a separate 0/1 column for each category.
# its more columns but the model doesnt assume any ordering between them.
# ============================================================

x = df.drop(columns=['price'], axis=1)  # everything except price is input
y = df['price']                          # price is what we're predicting

# get_dummies handles model, transmission, fuelType — drop_first removes one
# column per group to avoid the dummy variable trap
x_one_encoded = pd.get_dummies(x, columns=['model', 'transmission', 'fuelType'], drop_first=True).astype(int)
# print(x_one_encoded)
# ============================================================
# APPROACH 2: Label Encoding (same data, different method)
# instead of making separate columns, label encoding just replaces
# each category with a number — Fiesta=5, Focus=6 etc.
# faster and fewer columns but the model might think Focus > Fiesta
# just because the number is higher, which isnt meaningful here.
# ============================================================

encoder = LabelEncoder()
columns = ['model', 'transmission', 'fuelType']
xlable = x

for i in columns:
    xlable[i] = encoder.fit_transform(xlable[i])
# print(xlable)
#        model  year  transmission  mileage  fuelType  tax   mpg  engineSize
# 0          5  2017             0    15944         4  150  57.7         1.0
# 1          6  2018             1     9083         4  150  57.7         1.0
# ...etc.

# scale the numeric columns so nothing dominates just because its a big number
# year, mileage, tax etc. are all on very different scales without this
numeric_columns = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()

x_one_encoded[numeric_columns] = scaler.fit_transform(x_one_encoded[numeric_columns])
# print(x_one_encoded)  — numbers now centred around 0

# scale the label-encoded version too — same reason
# all columns this time since label encoding kept them all in one place
xlable[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
        'engineSize']] = scaler.fit_transform(
    xlable[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg',
            'engineSize']])
# print(xlable)  — everything scaled now

# ============================================================
# MODEL 1 — trained on one-hot encoded data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(x_one_encoded, y, test_size=0.20, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_predictions = model.predict(X_test)
# print(y_predictions)
# [ 6866.36  9347.68  9362.18 ... 31454.95  9781.20 15253.50]

# print(y_test) — predicted values are very close to the actual prices
# 17610     6995
# 7076      8999
# 1713      7998

r2 = r2_score(y_test, y_predictions)
# print(r2)
# 0.8464  — model got about 84% right

# adjusted R² — same as R² but drops if useless columns are added
# both numbers being close means the columns we used are actually helpful
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
# print(adjusted_r2)
# 0.8450

# ============================================================
# MODEL 2 — same model, but trained on label-encoded data instead
# doing this to compare — is one encoding method actually better?
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(xlable, y, test_size=0.20, random_state=42)
model2 = LinearRegression()
model2.fit(X_train, y_train)

y_predictions = model2.predict(X_test)
# print(y_predictions)
# [ 6152.08  9374.39  9464.92 ... 21787.05  9776.44 15806.25]

# print(y_test)
# 17610     6995
# 7076      8999
# notice the predictions are a bit further off compared to model 1

r2_model2 = r2_score(y_test, y_predictions)
# print(r2_model2)
# 0.7366  — dropped to 73%, that's a noticeable fall

# conclusion:
# one-hot encoding: 84%
# label encoding:   73%
# the 11% gap shows that label encoding hurt the model here — it made the model
# treat car models like they have an order (Fiesta < Focus < Galaxy etc.)
# which isnt true. one-hot avoids that by giving each model its own column.
