# ------importing libraries necessary for project-------
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')  # to ignore warnings due to depricated packages if any

df = pd.read_csv('heart.csv')  # reading data
print(df.head())

# ------starting with exploratory data analysis (EDA)------
# before touching anything, I just want to see what the data looks like
# checking column names, types, missing values etc — standard stuff at the start
print(df.columns)
# ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
#        'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
#        'HeartDisease']

print(df.shape)
print(df.info())
# Data columns (total 12 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   Age             918 non-null    int64
#  1   Sex             918 non-null    object
#  2   ChestPainType   918 non-null    object
#  3   RestingBP       918 non-null    int64
#  4   Cholesterol     918 non-null    int64
#  5   FastingBS       918 non-null    int64
#  6   RestingECG      918 non-null    object
#  7   MaxHR           918 non-null    int64
#  8   ExerciseAngina  918 non-null    object
#  9   Oldpeak         918 non-null    float64
#  10  ST_Slope        918 non-null    object
#  11  HeartDisease    918 non-null    int64
# dtypes: float64(1), int64(6), object(5)
print("null values: \n", df.isnull().sum())
#  Age               0
# Sex               0
# ChestPainType     0
# RestingBP         0
# Cholesterol       0
# FastingBS         0
# RestingECG        0
# MaxHR             0
# ExerciseAngina    0
# Oldpeak           0
# ST_Slope          0
# HeartDisease      0
# dtype: int64
# there are no null values

print(df.describe())
#               Age   RestingBP  ...     Oldpeak  HeartDisease
# count  918.000000  918.000000  ...  918.000000    918.000000
# mean    53.510893  132.396514  ...    0.887364      0.553377
# std      9.432617   18.514154  ...    1.066570      0.497414
# min     28.000000    0.000000  ...   -2.600000      0.000000
# 25%     47.000000  120.000000  ...    0.000000      0.000000
# 50%     54.000000  130.000000  ...    0.600000      1.000000
# 75%     60.000000  140.000000  ...    1.500000      1.000000
# max     77.000000  200.000000  ...    6.200000      1.000000

print(df.duplicated().sum())

# plotting graphs to get a better idea about the data
# just visually poking around before any cleaning — helps spot weird values

print(df['HeartDisease'].value_counts())


# HeartDisease
# 1    508
# 0    410
# Name: count, dtype: int64
# print(df['HeartDisease'].value_counts().plot(kind='bar'))
# plt.show()

# helper function — pass any column and a subplot position (1-4) and it draws a distribution chart
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(var, kde=True)


# plotting(df['Age'],1)
# plotting(df['RestingBP'],2)
# plotting(df['Cholesterol'],3)
# plotting(df['MaxHR'],4)
# plt.tight_layout()
# plt.show()
#
# sns.histplot(x= df['Cholesterol'])
# plt.title("cholesterol cannot be zero")
# plt.show()

# --------there are issues in data regarding cholestrol and blood pressure.
# blood pressure and cholesterol cannot be 0 — those are clearly bad data entries
# fix: calculate the mean of all valid (non-zero) values and plug that in instead

cholesterol_mean = round(df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean(), 2)
# round(cholesterol_mean)
print("Chelosterol mean:", cholesterol_mean)
df['Cholesterol'] = df['Cholesterol'].replace(0, cholesterol_mean)

# sns.histplot(x= df['Cholesterol'])
# plt.show()

print(df['Cholesterol'].value_counts())
# Cholesterol
# 244.64    172
# 254.00     11
# 220.00     10
# 223.00     10
# 204.00      9
#          ...
# 353.00      1
# 278.00      1
# 157.00      1
# 176.00      1
# 131.00      1

RestingBP_mean = round(df.loc[df['RestingBP'] != 0, 'RestingBP'].mean(), 2)
print("RestingBP", RestingBP_mean)
df['RestingBP'] = df['RestingBP'].replace(0, RestingBP_mean)
# sns.histplot(df['RestingBP'])
# plt.show()


# plotting(df['Age'],1)
# plotting(df['RestingBP'],2)
# plotting(df['Cholesterol'],3)
# plotting(df['MaxHR'],4)
# plt.tight_layout()
# plt.show()

# sns.countplot(x = df['Sex'],hue = df['HeartDisease'])
# plt.show()

columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
           'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',
           'HeartDisease']

# for col in columns:
#     sns.boxplot(x=df[col])
#     plt.show()

# sns.boxplot(x= df['HeartDisease'] , y=df['Cholesterol'], data=df)
# sns.violinplot(x= df['HeartDisease'] , y=df['Age'], data=df)
# plt.show()

# sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()

# --------------data processing and cleaning-------------

# the model only understands numbers — so converting all text columns to 0/1 using one-hot encoding
# drop_first=True removes one column per group to avoid redundancy (if all others are 0, it must be the dropped one)
df_encode = pd.get_dummies(df,drop_first=True)
df_encode = df_encode.astype(int)
print(df_encode)

# scaling the numeric columns so columns with big numbers don't unfairly dominate the model
# e.g. Age=55 vs Cholesterol=240 — without scaling the model might treat cholesterol as more important just because the number is bigger
numeric_columns = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
scaler = StandardScaler()
df_encode[numeric_columns] = scaler.fit_transform(df_encode[numeric_columns])

print(df_encode.head(20))
