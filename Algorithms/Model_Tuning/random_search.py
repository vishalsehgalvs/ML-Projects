# =============================================================================
# Randomized Search CV — Faster Hyperparameter Search
# =============================================================================
# Goal   : Same idea as Grid Search CV — find the best model settings.
#          The difference: instead of trying EVERY combination, it randomly
#          picks a few. n_iter controls how many combos to try.
#          Much faster when the param grid is large.
# Dataset: Iris — same dataset as Grid_Search_CV.py.
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

# Using the Iris dataset — built into seaborn, so no CSV needed.
# 3 species to classify (setosa, versicolor, virginica) — a multi-class problem, not just yes/no.

df = sns.load_dataset('iris')
# print(df.head())
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa

# No cleaning needed — Iris has no missing values, no text columns, no duplicates. Straight to the model.

# print(df['species'].unique())
# ['setosa' 'versicolor' 'virginica']

# X = the 4 flower measurements (inputs), y = species label (what we want to predict)

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Starting with a basic train/test split — to show the manual approach before we automate it.


model_KNN = KNeighborsClassifier(n_neighbors=13)
model_KNN.fit(X_train, y_train)  # model trained here
# .score() gives accuracy directly — no need to call predict() separately
KNN_score = model_KNN.score(X_test, y_test)
# print(KNN_score)
# 0.98 score with n_neighbors=5
# -------------------------------
# print(KNN_score)
# 1.0 score with n_neighbors=13, but this means the model is overfitting
# changing the hyperparameters affect the model performance

# Now trying SVM — same manual approach — to show that picking settings by hand is unpredictable.

# model_SVM = SVC(gamma='auto', C=10, kernel='linear')
model_SVM = SVC()
model_SVM.fit(X_train, y_train)
score_svm = model_SVM.score(X_test, y_test)
# print(score_svm) with C=30 and kernel rbf: 0.98
# print(score_svm) with C=10 and kernel linear: 0.98
# --------------------------------
# Manually changing settings is getting tedious and unreliable.
# Randomized Search CV solves this: give it a list of options, tell it
# how many combos to try (n_iter), and it randomly picks them, tests each
# with cross-validation, and returns the best one.
# Much faster than Grid Search when there are lots of combinations.

# Same structure as GridSearchCV — the only new thing is n_iter.
# n_iter=4 means: don't try all 8 combos, just randomly pick 4.
# Trade-off: faster, but might miss the absolute best by chance (rarely matters in practice).
random_classifier_svm = RandomizedSearchCV((model_SVM),{
    "C": [1, 10, 20, 30],
    "kernel": ['rbf', 'linear'],
}, cv=5, return_train_score=False, n_iter=4)
random_classifier_svm.fit(X, y)
random_classifier_svm_result = random_classifier_svm.cv_results_
# print(random_classifier_svm_result)
result = pd.DataFrame(random_classifier_svm_result)
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.max_rows', None)      # show all rows
pd.set_option('display.width', None)         # avoid wrapping
# print(result)
#    mean_fit_time  std_fit_time  mean_score_time  std_score_time param_kernel  \
# 0       0.001401      0.000489         0.001017        0.000337       linear
# 1       0.001090      0.000491         0.000802        0.000401       linear
# 2       0.001503      0.000637         0.000800        0.000400          rbf
# 3       0.001363      0.000366         0.000802        0.000401       linear
#
#    param_C                         params  split0_test_score  \
# 0       30  {'kernel': 'linear', 'C': 30}           1.000000
# 1       20  {'kernel': 'linear', 'C': 20}           1.000000
# 2       20     {'kernel': 'rbf', 'C': 20}           0.966667
# 3        1   {'kernel': 'linear', 'C': 1}           0.966667
#
#    split1_test_score  split2_test_score  split3_test_score  split4_test_score  \
# 0                1.0           0.900000           0.900000                1.0
# 1                1.0           0.900000           0.933333                1.0
# 2                1.0           0.966667           0.966667                1.0
# 3                1.0           0.966667           0.966667                1.0
#
#    mean_test_score  std_test_score  rank_test_score
# 0         0.960000        0.048990                4
# 1         0.966667        0.042164                3
# 2         0.980000        0.016330                1
# 3         0.980000        0.016330                1

# show only important information
clean_results = result[['param_C', 'param_kernel', 'mean_test_score', 'rank_test_score']]
# print(clean_results.sort_values('rank_test_score'))
#
#    param_C param_kernel  mean_test_score  rank_test_score
# 1        1       linear         0.980000                1
# 0       10       linear         0.973333                2
# 2       30          rbf         0.973333                2
# 3       20       linear         0.966667                4

classifier_KNN = RandomizedSearchCV((model_KNN), {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean']
}, cv=5, return_train_score=False,n_iter=5)
classifier_KNN.fit(X, y)
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # avoid wrapping
result_df_KNN = pd.DataFrame(classifier_KNN.cv_results_)
# print(result_df_KNN)
#    mean_fit_time  std_fit_time  mean_score_time  std_score_time param_weights  param_n_neighbors param_metric                                             params  split0_test_score  split1_test_score  split2_test_score  split3_test_score  split4_test_score  mean_test_score  std_test_score  rank_test_score
# 0       0.001900      0.000197         0.001201        0.000402      distance                 13    minkowski  {'weights': 'distance', 'n_neighbors': 13, 'me...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                2
# 1       0.001630      0.000518         0.000980        0.000032      distance                  3    euclidean  {'weights': 'distance', 'n_neighbors': 3, 'met...           0.966667           0.966667           0.933333           0.966667                1.0         0.966667        0.021082                5
# 2       0.001406      0.000503         0.000983        0.000023      distance                 11    euclidean  {'weights': 'distance', 'n_neighbors': 11, 'me...           0.966667           1.000000           1.000000           0.966667                1.0         0.986667        0.016330                1
# 3       0.001586      0.000505         0.001006        0.000011      distance                  9    euclidean  {'weights': 'distance', 'n_neighbors': 9, 'met...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944                3
# 4       0.001005      0.000015         0.002198        0.000388       uniform                 13    minkowski  {'weights': 'uniform', 'n_neighbors': 13, 'met...           0.933333           1.000000           0.966667           0.966667                1.0         0.973333        0.024944                3


# show only important information
clean_results_KNN = result_df_KNN[['param_n_neighbors','param_weights','param_metric','mean_test_score','rank_test_score']]
# print(clean_results_KNN.sort_values('rank_test_score'))
#    param_n_neighbors param_weights param_metric  mean_test_score  rank_test_score
# 1                  7      distance    minkowski         0.980000                1
# 0                 15      distance    euclidean         0.973333                2
# 3                  5       uniform    euclidean         0.973333                2
# 4                  9       uniform    euclidean         0.973333                2
# 2                  5      distance    euclidean         0.966667                5