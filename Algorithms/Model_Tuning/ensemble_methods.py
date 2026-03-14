# =============================================================================
# Ensemble Methods — Stacking, Bagging, Boosting and XGBoost
# =============================================================================
# Goal   : Show all three ensemble approaches on the same Iris dataset so
#          results are directly comparable.
# Dataset: Iris — built into seaborn, no CSV needed.
#          150 flowers, 4 measurements, 3 species to predict.
# =============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Loading Iris — same dataset used in Grid Search and Random Search examples.
df = sns.load_dataset('iris')
# print(df.head())
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa

# X = the 4 flower measurements, y = species (what we're predicting)
X = df.drop('species', axis=1)
y = df['species']
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# STACKING
# =============================================================================
# Step 1: define the base learners — three different model types.
# Each one sees the data differently and brings a different perspective.
# Using probability=True on SVC so it outputs probabilities, not just 0/1.
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svc', SVC(probability=True, kernel='rbf', random_state=42)),
    ('lr', LogisticRegression(max_iter=1000))
]

# Step 2: the meta-learner (final decision maker).
# It takes the base learners' predictions as its input and makes the final call.
mera_learner = LogisticRegression(max_iter=1000)

# Step 3: put it all together.
# cv=5 means each base learner is trained using 5-fold cross-validation
# so the meta-learner is trained on predictions the base models haven't seen before.
stacking_classifier = StackingClassifier(
    estimators=base_learners,
    final_estimator=mera_learner,
    cv=5)

stacking_classifier.fit(X_train, y_train)
y_predictions = stacking_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)
# print(accuracy)

# =============================================================================
# BAGGING — Random Forest
# =============================================================================
# Builds 100 decision trees, each trained on a random subset of the data.
# Final prediction = majority vote across all 100 trees.
# Much more stable than a single decision tree.
rf_model = RandomForestClassifier(
    n_estimators=100,    # build 100 trees and take a majority vote
    max_depth=None,      # let each tree grow as deep as it wants
    random_state=42
)
rf_model.fit(X_train, y_train)
y_predictions_rf_model = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_predictions)
# print(accuracy)

# =============================================================================
# BOOSTING — AdaBoost and Gradient Boosting
# =============================================================================
# Unlike bagging (parallel trees), boosting trains models one after another.
# Each new model pays extra attention to the rows the previous model got wrong.
# They learn from mistakes sequentially, getting better with each round.

# AdaBoost — original boosting algorithm.
# Misclassified rows get higher weight so the next model focuses on them more.
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_boost.fit(X_train, y_train)
ada_prediction = ada_boost.predict(X_test)
accuracy_ada = accuracy_score(y_test, ada_prediction)
# print(accuracy_ada)
# 0.9333333333333333

# Gradient Boosting — more powerful than AdaBoost.
# Builds trees that correct the residual errors of the previous tree.
# learning_rate controls how much each new tree corrects the previous mistake.
# Smaller rate = slower learning but often better final result.
gradient_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gradient_boosting.fit(X_train, y_train)
gradient_prediction = gradient_boosting.predict(X_test)
accuracy_gradient_boosting = accuracy_score(y_test, gradient_prediction)
# print(accuracy_gradient_boosting)
# 1.0

# =============================================================================
# XGBoost — Extreme Gradient Boosting
# =============================================================================
# The most widely used boosting algorithm in competitions and real-world work.
# Faster and more powerful than standard Gradient Boosting.
#
# XGBoost requires numeric labels — it can't handle strings like 'setosa'.
# That's why we re-encode y here: setosa=0, versicolor=1, virginica=2.
# The plain y (strings) worked fine for sklearn models above, but XGBoost
# will throw an error if you pass it text labels.
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False,
                          eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train,y_train)
xgb_predict = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test,xgb_predict)
# print(xgb_accuracy)
# 1.0

