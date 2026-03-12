# Decision Tree — Titanic Survival Prediction

Fourth algorithm in the classification series. Same Titanic dataset and cleaning pipeline. Algorithm changed to `DecisionTreeClassifier`.

The core idea is different from everything before: instead of drawing a line or calculating probabilities, a decision tree builds a flowchart. It asks a sequence of yes/no questions about each passenger and follows the branches to a prediction.

---

## Dataset

Titanic from seaborn — no CSV needed. 891 passengers, 15 columns, cleaned to 889 rows and 8 features.

Dropped: `deck`, `alive`, `class`, `embark_town`, `who`, `adult_male`.
Missing: 177 age values filled with column mean. 2 embarked rows dropped.
Encoded: sex (female=0, male=1), embarked (C=0, Q=1, S=2), booleans to int.

---

## Results

```
Accuracy: 77.0%

Confusion Matrix:
[[88  21]
 [20  49]]

Classification Report:
              precision    recall    f1
   Died (0)     0.81       0.81     0.81
Survived (1)   0.70       0.71     0.71
```

All four classifiers on this dataset:

|                 | Logistic Regression | KNN K=5 | Naive Bayes | Decision Tree |
| --------------- | ------------------- | ------- | ----------- | ------------- |
| Accuracy        | **80.3%**           | 78.1%   | 77.5%       | 77.0%         |
| Survived recall | 0.77                | 0.71    | 0.78        | 0.71          |

Comes in last on accuracy, but no `max_depth` was set — the tree is fully grown and likely overfitting. Tuning it would probably close the gap.

---

## Run it

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
python decision_tree_algorithm.py
```

Results are commented-out print statements in the script. Uncomment to see output.

---

_Part of the Algorithms/Classification series._
