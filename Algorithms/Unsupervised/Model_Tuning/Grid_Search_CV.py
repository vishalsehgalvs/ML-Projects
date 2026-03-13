# =============================================================================
# Grid Search CV — Finding the Best Hyperparameters Automatically
# =============================================================================
# Goal   : Show why manually picking model settings is unreliable, then use
#          GridSearchCV to try every combination automatically and find the best.
# Dataset: Iris — built into seaborn, no CSV needed.
#          3 flower species (setosa, versicolor, virginica), 150 rows, 4 columns.
#          Already clean — no missing values, no text to encode.
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
from sklearn.model_selection import GridSearchCV

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

# Now trying SVM with manually picked settings — same point: different settings, different scores.

model_SVM = SVC(gamma='auto', C=10, kernel='linear')
model_SVM.fit(X_train, y_train)
score_svm = model_SVM.score(X_test, y_test)
# print(score_svm) with cv =30 and kernel as rbf
# 0.98
# print(score_svm) with cv as 10 and kernel as linear
# 0.98
# --------------------------------
# Manually changing settings is getting tedious and unreliable.
# Too many combinations to try by hand, and no way to know which is best.
# GridSearchCV fixes this — it tries every combination automatically,
# runs cross-validation on each one, and returns the winner.

classifier = GridSearchCV((model_SVM), param_grid={
    "C": [1, 10, 20, 30],
    "kernel": ['rbf', 'linear']
}, cv=5, return_train_score=False)
classifier.fit(X, y)  # pass the full dataset — GridSearchCV handles the train/test splits internally
# print(classifier.cv_results_)  # raw output is a messy dict — convert to DataFrame to read it
result_df = pd.DataFrame(classifier.cv_results_)
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # avoid wrapping
# print(result_df)
#    mean_fit_time  std_fit_time  mean_score_time  std_score_time  param_C param_kernel                         params  split0_test_score  split1_test_score  split2_test_score  split3_test_score  split4_test_score  mean_test_score  std_test_score  rank_test_score
# 0       0.002000      0.000001         0.001000    4.101908e-07        1          rbf      {'C': 1, 'kernel': 'rbf'}           0.966667                1.0           0.966667           0.966667                1.0         0.980000        0.016330                1
# 1       0.001398      0.000490         0.001201    3.996163e-04        1       linear   {'C': 1, 'kernel': 'linear'}           0.966667                1.0           0.966667           0.966667                1.0         0.980000        0.016330                1
# 2       0.001601      0.000491         0.001199    4.004269e-04       10          rbf     {'C': 10, 'kernel': 'rbf'}           0.966667                1.0           0.966667           0.966667                1.0         0.980000        0.016330                1
# 3       0.001399      0.000491         0.001000    2.336015e-07       10       linear  {'C': 10, 'kernel': 'linear'}           1.000000                1.0           0.900000           0.966667                1.0         0.973333        0.038873                4
# 4       0.001805      0.000403         0.000998    2.065316e-06       20          rbf     {'C': 20, 'kernel': 'rbf'}           0.966667                1.0           0.900000           0.966667                1.0         0.966667        0.036515                5
# 5       0.001602      0.000486         0.000801    4.003541e-04       20       linear  {'C': 20, 'kernel': 'linear'}           1.000000                1.0           0.900000           0.933333                1.0         0.966667        0.042164                6
# 6       0.001200      0.000400         0.001201    4.000687e-04       30          rbf     {'C': 30, 'kernel': 'rbf'}           0.966667                1.0           0.900000           0.933333                1.0         0.960000        0.038873                7
# 7       0.001400      0.000491         0.001000    1.257983e-06       30       linear  {'C': 30, 'kernel': 'linear'}           1.000000                1.0           0.900000           0.900000                1.0         0.960000        0.048990                7

# show only important information
# clean_results = result_df[['param_C', 'param_kernel', 'mean_test_score', 'rank_test_score']]
# print(clean_results.sort_values('rank_test_score'))
#    param_C param_kernel  mean_test_score  rank_test_score
# 0        1          rbf         0.980000                1
# 1        1       linear         0.980000                1
# 2       10          rbf         0.980000                1
# 3       10       linear         0.973333                4
# 4       20          rbf         0.966667                5
# 5       20       linear         0.966667                6
# 6       30          rbf         0.960000                7
# 7       30       linear         0.960000                7

# Best setting found: C=1 with rbf (or linear) — lower C = more relaxed boundary = less overfitting.
# --------------- Now doing the same Grid Search for KNN --------------------------------
# KNeighborsClassifier params: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# n_neighbors: Any = 5,
#              *,
#              weights: Any = "uniform",
#              algorithm: Any = "auto",
#              leaf_size: Any = 30,
#              p: Any = 2,
#              metric: Any = "minkowski",
#              metric_params: Any = None,
#              n_jobs: Any = None) -> None
classifier_KNN = GridSearchCV((model_KNN), param_grid={
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean']
}, cv=5, return_train_score=False)
classifier_KNN.fit(X,y)
pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # avoid wrapping
result_df_KNN = pd.DataFrame(classifier_KNN.cv_results_)
# print(result_df_KNN)
#     mean_fit_time  std_fit_time  mean_score_time  std_score_time param_metric  param_n_neighbors param_weights                                             params  split0_test_score  split1_test_score  split2_test_score  split3_test_score  split4_test_score  mean_test_score  std_test_score  rank_test_score
# 0        0.000803      0.000401         0.002809        0.000516    minkowski                  3       uniform  {'metric': 'minkowski', 'n_neighbors': 3, 'wei...           0.966667           0.966667           0.933333           0.966667                1.0         0.966667        0.021082               21
# 1        0.001001      0.000008         0.001303        0.000402    minkowski                  3      distance  {'metric': 'minkowski', 'n_neighbors': 3, 'wei...           0.966667           0.966667           0.933333           0.966667                1.0         0.966667        0.021082               21
# 2        0.001171      0.000420         0.002523        0.000319    minkowski                  5       uniform  {'metric': 'minkowski', 'n_neighbors': 5, 'wei...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944               11
# 3        0.001000      0.000006         0.001399        0.000372    minkowski                  5      distance  {'metric': 'minkowski', 'n_neighbors': 5, 'wei...           0.966667           1.000000           0.900000           0.966667                1.0         0.966667        0.036515               21
# 4        0.001003      0.000006         0.002707        0.000397    minkowski                  7       uniform  {'metric': 'minkowski', 'n_neighbors': 7, 'wei...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                3
# 5        0.001508      0.000642         0.001504        0.000453    minkowski                  7      distance  {'metric': 'minkowski', 'n_neighbors': 7, 'wei...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                3
# 6        0.001300      0.000408         0.002469        0.000409    minkowski                  9       uniform  {'metric': 'minkowski', 'n_neighbors': 9, 'wei...           0.966667           1.000000           0.966667           0.933333                1.0         0.973333        0.024944               11
# 7        0.001321      0.000412         0.000986        0.000028    minkowski                  9      distance  {'metric': 'minkowski', 'n_neighbors': 9, 'wei...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944               11
# 8        0.001406      0.000498         0.002314        0.000256    minkowski                 11       uniform  {'metric': 'minkowski', 'n_neighbors': 11, 'we...           0.933333           1.000000           1.000000           0.966667                1.0         0.980000        0.026667                3
# 9        0.001117      0.000232         0.001293        0.000410    minkowski                 11      distance  {'metric': 'minkowski', 'n_neighbors': 11, 'we...           0.966667           1.000000           1.000000           0.966667                1.0         0.986667        0.016330                1
# 10       0.000803      0.000402         0.002307        0.000611    minkowski                 13       uniform  {'metric': 'minkowski', 'n_neighbors': 13, 'we...           0.933333           1.000000           0.966667           0.966667                1.0         0.973333        0.024944               11
# 11       0.001152      0.000491         0.001511        0.000449    minkowski                 13      distance  {'metric': 'minkowski', 'n_neighbors': 13, 'we...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                3
# 12       0.001188      0.000251         0.002688        0.000385    minkowski                 15       uniform  {'metric': 'minkowski', 'n_neighbors': 15, 'we...           0.933333           1.000000           0.933333           0.966667                1.0         0.966667        0.029814               21
# 13       0.001401      0.000489         0.000905        0.000190    minkowski                 15      distance  {'metric': 'minkowski', 'n_neighbors': 15, 'we...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944               11
# 14       0.001385      0.000511         0.002166        0.000400    euclidean                  3       uniform  {'metric': 'euclidean', 'n_neighbors': 3, 'wei...           0.966667           0.966667           0.933333           0.966667                1.0         0.966667        0.021082               21
# 15       0.001004      0.000006         0.001503        0.000446    euclidean                  3      distance  {'metric': 'euclidean', 'n_neighbors': 3, 'wei...           0.966667           0.966667           0.933333           0.966667                1.0         0.966667        0.021082               21
# 16       0.001331      0.000415         0.002179        0.000415    euclidean                  5       uniform  {'metric': 'euclidean', 'n_neighbors': 5, 'wei...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944               11
# 17       0.001307      0.000386         0.001504        0.000430    euclidean                  5      distance  {'metric': 'euclidean', 'n_neighbors': 5, 'wei...           0.966667           1.000000           0.900000           0.966667                1.0         0.966667        0.036515               21
# 18       0.001702      0.000397         0.002307        0.000607    euclidean                  7       uniform  {'metric': 'euclidean', 'n_neighbors': 7, 'wei...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                3
# 19       0.001205      0.000400         0.001718        0.000377    euclidean                  7      distance  {'metric': 'euclidean', 'n_neighbors': 7, 'wei...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                3
# 20       0.001120      0.000240         0.001913        0.000530    euclidean                  9       uniform  {'metric': 'euclidean', 'n_neighbors': 9, 'wei...           0.966667           1.000000           0.966667           0.933333                1.0         0.973333        0.024944               11
# 21       0.001340      0.000422         0.001381        0.000788    euclidean                  9      distance  {'metric': 'euclidean', 'n_neighbors': 9, 'wei...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944               11
# 22       0.001002      0.000003         0.002203        0.000245    euclidean                 11       uniform  {'metric': 'euclidean', 'n_neighbors': 11, 'we...           0.933333           1.000000           1.000000           0.966667                1.0         0.980000        0.026667                3
# 23       0.001001      0.000002         0.001401        0.000489    euclidean                 11      distance  {'metric': 'euclidean', 'n_neighbors': 11, 'we...           0.966667           1.000000           1.000000           0.966667                1.0         0.986667        0.016330                1
# 24       0.001202      0.000247         0.002204        0.000398    euclidean                 13       uniform  {'metric': 'euclidean', 'n_neighbors': 13, 'we...           0.933333           1.000000           0.966667           0.966667                1.0         0.973333        0.024944               11
# 25       0.000801      0.000401         0.001518        0.000449    euclidean                 13      distance  {'metric': 'euclidean', 'n_neighbors': 13, 'we...           0.966667           1.000000           0.966667           0.966667                1.0         0.980000        0.016330                3
# 26       0.001281      0.000417         0.002125        0.000242    euclidean                 15       uniform  {'metric': 'euclidean', 'n_neighbors': 15, 'we...           0.933333           1.000000           0.933333           0.966667                1.0         0.966667        0.029814               21
# 27       0.001236      0.000286         0.001170        0.000416    euclidean                 15      distance  {'metric': 'euclidean', 'n_neighbors': 15, 'we...           0.966667           1.000000           0.933333           0.966667                1.0         0.973333        0.024944               11

# show only important information
clean_results_KNN = result_df_KNN[['param_n_neighbors','param_weights','param_metric','mean_test_score','rank_test_score']]
print(clean_results_KNN.sort_values('rank_test_score'))
#     param_n_neighbors param_weights param_metric  mean_test_score  rank_test_score
# 9                  11      distance    minkowski         0.986667                1
# 23                 11      distance    euclidean         0.986667                1
# 8                  11       uniform    minkowski         0.980000                3
# 5                   7      distance    minkowski         0.980000                3
# 11                 13      distance    minkowski         0.980000                3
# 22                 11       uniform    euclidean         0.980000                3
# 19                  7      distance    euclidean         0.980000                3
# 4                   7       uniform    minkowski         0.980000                3
# 18                  7       uniform    euclidean         0.980000                3
# 25                 13      distance    euclidean         0.980000                3
# 21                  9      distance    euclidean         0.973333               11
# 10                 13       uniform    minkowski         0.973333               11
# 13                 15      distance    minkowski         0.973333               11
# 7                   9      distance    minkowski         0.973333               11
# 2                   5       uniform    minkowski         0.973333               11
# 6                   9       uniform    minkowski         0.973333               11
# 27                 15      distance    euclidean         0.973333               11
# 24                 13       uniform    euclidean         0.973333               11
# 20                  9       uniform    euclidean         0.973333               11
# 16                  5       uniform    euclidean         0.973333               11
# 0                   3       uniform    minkowski         0.966667               21
# 3                   5      distance    minkowski         0.966667               21
# 12                 15       uniform    minkowski         0.966667               21
# 1                   3      distance    minkowski         0.966667               21
# 15                  3      distance    euclidean         0.966667               21
# 14                  3       uniform    euclidean         0.966667               21
# 17                  5      distance    euclidean         0.966667               21
# 26                 15       uniform    euclidean         0.966667               21
