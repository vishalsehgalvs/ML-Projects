# Support Vector Machine — Titanic Survival Prediction

## What it does

Finds the boundary that separates survived vs died with the **widest possible gap** on either side (maximum margin). Uses the RBF kernel to draw curved, non-linear boundaries.

## Dataset

Titanic — 891 rows, 15 columns  
After cleaning: 889 rows, 8 features  
Features used: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`, `alone`

## Results

```
Accuracy:  82.6%   ← Best of all 5 algorithms

Confusion Matrix:
[[96  13]
 [18  51]]

Classification Report:
              precision  recall  f1-score
   Died (0)     0.84     0.88     0.86
Survived (1)    0.80     0.74     0.77
```

## Algorithm comparison

| Algorithm           | Accuracy  |
| ------------------- | --------- |
| Logistic Regression | 80.3%     |
| KNN K=5             | 78.1%     |
| Naive Bayes         | 77.5%     |
| Decision Tree       | 77.0%     |
| **SVM (RBF)**       | **82.6%** |

## Run

```bash
python support_vector_machine.py
```

_Requires: pandas, numpy, scikit-learn, seaborn_
