# =============================================================================
# Insurance Charges Prediction - ML Project (First ML Project)
# Dataset Source: Kaggle
# Goal: Analyse insurance data, preprocess it, and select important features
#       that influence insurance charges.
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Load & Explore the Raw Dataset
# =============================================================================

df = pd.read_csv('insurance.csv')  # insurance data set has been taken from kaggle

# Full dataset overview
print(df)
print("=============================")

# First 5 rows — quick sanity check
print(df.head())
print("=============================")

# Column names, dtypes, and non-null counts
print(df.info())
print("=============================")

# Summary statistics: mean, std, min, max, quartiles
print(df.describe())
print("=============================")

# Check for missing values in each column
print(df.isnull().sum())
print("=============================")

# List all column names
print(df.columns)
# =============================================================================
# STEP 2: Exploratory Data Analysis (EDA)
# Plots are commented out — uncomment individually to view each visualisation.
# =============================================================================

numeric_columns = ['age', 'bmi', 'children', 'charges']  # only numeric columns need distribution check

# ── Distribution plots (histogram + KDE) for all numeric features ─────────────
# for col in numeric_columns:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[col], kde=True, bins=20)
#     plt.show()

# ── Count of smokers vs non-smokers ──────────────────────────────────────────
# sns.countplot(x=df['smoker'])
# plt.show()

# ── Boxplots to identify outliers in numeric features ────────────────────────
# for col in numeric_columns:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=df[col])
# plt.show()

print("=============================")

# ── Correlation heatmap (numeric columns only) ────────────────────────────────
# numeric_only=True is optional — heatmap works only with numeric values anyway
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

print("")  # new line

# =============================================================================
# STEP 3: Data Cleaning & Preprocessing
# =============================================================================

print("=============================")
print("Cleaning and preprocessing the data")
print("=============================")

# Work on a copy to preserve the original raw dataframe
df_cleaned = df.copy()
print(df_cleaned.head())

# ── Remove duplicate rows ──────────────────────────────────────────────────────
print("before dropping duplicates", df_cleaned.shape)  # (1338, 7)
df_cleaned.drop_duplicates(inplace=True)
print("After dropping duplicates", df_cleaned.shape)   # (1337, 7)

# ── Confirm no null/missing values remain ─────────────────────────────────────
print(df_cleaned.isnull().sum())
# age         0
# sex         0
# bmi         0
# children    0
# smoker      0
# region      0
# charges     0
# dtype: int64
# no null value in our case

# ── Check dtypes — object columns must be encoded to numeric ──────────────────
print(df_cleaned.dtypes)
# age           int64
# sex          object
# bmi         float64
# children      int64
# smoker       object
# region       object
# charges     float64
# dtype: object
# all objects (strings) have to be converted into numeric values

# Count of male vs female before encoding
print(df_cleaned['sex'].value_counts())
# sex
# male      675
# female    662
# Name: count, dtype: int64

# ── Label Encoding: binary categorical columns ────────────────────────────────
# male → 0, female → 1
df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
# yes → 1, no → 0
df_cleaned['smoker'] = df_cleaned['smoker'].map({'yes': 1, 'no': 0})
print(df_cleaned.head)

# ── Rename columns to reflect the encoded meaning ─────────────────────────────
df_cleaned.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)

print(df_cleaned.head())

# Count of each region before one-hot encoding
print(df_cleaned['region'].value_counts())
# region
# southeast    364
# southwest    325
# northwest    324
# northeast    324
# Name: count, dtype: int64
# these regions have to be encoded as well

# ── One-Hot Encoding: 'region' column (drop_first avoids dummy variable trap) ──
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
print(df_cleaned.head())  # regions are depicted as True/False — convert to int next

# Convert all boolean columns (from get_dummies) to integer (0 / 1)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.head())

# =============================================================================
# STEP 4: Feature Engineering — BMI Category
# Adding a domain-knowledge-based feature: BMI grouped into clinical categories
# =============================================================================

print("=============================")
print("Creating new data")
print("=============================")

# ── Optional: view BMI distribution before binning ────────────────────────────
# sns.histplot(df_cleaned['bmi'])
# plt.show()

# Bin BMI into clinical categories using standard WHO cut-offs
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['underweight', 'normal', 'overweight', 'obese'])

print(df_cleaned.head())

# One-hot encode the new BMI category (drop_first to avoid multicollinearity)
df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)
df_cleaned = df_cleaned.astype('int')  # convert boolean flags to integers

print(df_cleaned.columns)

# =============================================================================
# STEP 5: Feature Scaling
# Standardise continuous numeric columns so they have mean=0 and std=1.
# This prevents larger-scale features from dominating distance-based models.
# =============================================================================

cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
print(df_cleaned)


# =============================================================================
# STEP 6: Feature Selection — Pearson Correlation (Numeric Features)
# Pearson correlation measures linear relationship between each feature and
# the target variable (charges). Range: -1 (negative) to +1 (positive).
# =============================================================================

selected_features = ['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges',
                     'region_northwest', 'region_southeast', 'region_southwest',
                     'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese']

# Compute Pearson correlation of each feature against 'charges'
correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(list(correlations.items()), columns=['feature', 'Pearson correlation'])
correlation_df.sort_values(by="Pearson correlation", ascending=False)
print(correlation_df)

#                     feature  Pearson correlation
# 0                       age             0.298309
# 1                 is_female            -0.058046
# 2                       bmi             0.196236
# 3                  children             0.067390
# 4                 is_smoker             0.787234   ← strongest predictor
# 5                   charges             1.000000
# 6          region_northwest            -0.038695
# 7          region_southeast             0.073577
# 8          region_southwest            -0.043637
# 9       bmi_category_normal            -0.104042
# 10  bmi_category_overweight            -0.120601
# 11       bmi_category_obese             0.200348

# =============================================================================
# STEP 7: Feature Selection — Chi-Square Test (Categorical Features)
# Chi-square tests whether a categorical feature is statistically independent
# of the target. p-value < 0.05 → reject null → feature is significant (keep).
# =============================================================================

categorical_features = [
    'is_female', 'is_smoker', 'region_northwest', 'region_southeast',
    'region_southwest', 'bmi_category_normal', 'bmi_category_overweight',
    'bmi_category_obese'
]

alpha = 0.05  # significance threshold

# Bin charges into 4 equal-frequency buckets to create a categorical target
df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'], q=4, labels=False)

chi2_result = {}

for col in categorical_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = "Reject Null (Keep Features)" if p_val < alpha else "Accept Null ( Drop Feature)"
    chi2_result[col] = {
        'chi2_statistics': chi2_stat,
        'p_val': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_result).T
chi2_df = chi2_df.sort_values(by='p_val')
print(chi2_df)

#                         chi2_statistics     p_val                     Decision
# is_smoker                    848.219178       0.0  Reject Null (Keep Features)
# region_southeast              15.998167  0.001135  Reject Null (Keep Features)
# is_female                     10.258784   0.01649  Reject Null (Keep Features)
# bmi_category_obese             8.515711  0.036473  Reject Null (Keep Features)
# region_southwest               5.091893  0.165191  Accept Null ( Drop Feature)
# bmi_category_overweight         4.25149  0.235557  Accept Null ( Drop Feature)
# bmi_category_normal            3.708088   0.29476  Accept Null ( Drop Feature)
# region_northwest                1.13424  0.768815  Accept Null ( Drop Feature)

# =============================================================================
# STEP 8: Final Feature Set
# Keeping only the statistically significant features identified above.
# This is the clean, ready-to-model dataframe.
# =============================================================================

final_df = df_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker',
                        'charges', 'region_southeast', 'bmi_category_obese']]
print(final_df)