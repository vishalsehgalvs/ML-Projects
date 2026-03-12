# Naive Bayes — Titanic Survival Prediction

Third algorithm in the classification series. Same Titanic dataset, same cleaning, different algorithm.

This one is based on probability rather than drawing lines or measuring distances. The idea is simple — look at what you know about a passenger, figure out how likely they are to have survived vs died, go with whichever is higher.

---

## What's different here compared to KNN and Logistic Regression

The main thing worth noting: no feature scaling. With KNN, scaling was mandatory because KNN works by measuring how close two data points are to each other. If fare goes up to 500 and pclass only goes up to 3, fare dominates every distance calculation.

Naive Bayes doesn't measure distance at all. It builds probability distributions — one for each feature. Since every feature is treated independently, the size of the numbers doesn't affect anything. Scaling is imported in the code but intentionally skipped.

---

## The algorithm

For each new passenger, Naive Bayes calculates:

- probability this person survived, based on their age, sex, class, etc.
- probability this person died, based on the same things

Whichever is higher wins.

It figures out these probabilities by looking at the training data — splitting it into survived and died piles, then computing the average and spread of each feature in each pile. When a new passenger comes in, it checks how well their numbers fit each pile's distribution.

The "naive" part is that it assumes each feature is independent. So knowing someone paid a high fare doesn't influence what it thinks about their class or sex. That assumption is obviously wrong — rich passengers were more likely to be first class — but it still produces reasonable predictions. Weird but true.

---

## Why GaussianNB specifically

There are a few versions of Naive Bayes. Gaussian is for continuous numbers (age, fare, pclass). Multinomial is for word counts, which is why Naive Bayes is famous for spam filtering. Bernoulli is for binary yes/no features.

Since most of our features are continuous numbers, GaussianNB is the right pick.

---

## Dataset

Titanic — loaded directly from seaborn with `sns.load_dataset('titanic')`, no CSV file needed.

Started with 891 passengers, 15 columns. Cleaned down to 889 rows and 8 features.

Columns dropped: `deck` (mostly blank), `alive` (same as survived — cheating), `class` (same as pclass), `embark_town` (same as embarked), `who` and `adult_male` (already covered by sex and age).

Missing values: age filled with mean, 2 embarked rows dropped.

Text converted to numbers: sex (female=0, male=1), embarked (C=0, Q=1, S=2).

---

## Results

```
Accuracy: 77.5%

Confusion Matrix:
[[84  25]
 [15  54]]
```

Reading the matrix:

- 84 correct death predictions
- 54 correct survival predictions
- 25 false alarms (predicted survived, actually died)
- 15 missed survivors (predicted died, actually survived)

```
Classification Report:
              precision    recall    f1
   Died (0)     0.85       0.77     0.81
Survived (1)   0.68       0.78     0.73
```

Compared to the other two:

|                 | Logistic Regression | KNN K=5 | Naive Bayes |
| --------------- | ------------------- | ------- | ----------- |
| Accuracy        | 80.3%               | 78.1%   | 77.5%       |
| Survived recall | 0.77                | 0.71    | **0.78**    |

Logistic Regression wins on accuracy. But Naive Bayes has the best recall for survivors — it missed fewer actual survivors than either of the other two. All three are within 3% of each other, which made me realise the data cleaning step probably matters more than which algorithm you pick.

---

## Running it

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
python naive_bayes_algorithm.py
```

Results are in the script as commented-out print statements. Uncomment whatever you want to see.

---

## Files

```
Naive_Bayes_Algorithm/
├── naive_bayes_algorithm.py
├── NOTES.md
└── README.md
```

---

_Part of the Algorithms/Classification series._
