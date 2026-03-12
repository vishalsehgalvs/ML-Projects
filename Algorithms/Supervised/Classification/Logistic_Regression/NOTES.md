# Notes — Titanic Logistic Regression

Things I want to remember. Writing it down so I can come back later and know what I did and why.

---

## How this is different from the previous projects

In insurance and Ford I was guessing a number — charges, price. Could be anything: 12000, 8500, 54000.

Here the answer is always 0 or 1. Person survived or they didn't. So two things change:

- different model — Logistic Regression instead of Linear
- different way to check the score — confusion matrix instead of R²

---

## First look at the data

Dataset is baked into seaborn — `sns.load_dataset('titanic')`. 891 rows, 15 columns.

Ran `df.info()` and `df.isnull().sum()` to see what's there.

Three columns had blanks:

- `age` — 177 missing (about 1 in 5 passengers had no age listed)
- `deck` — 688 missing (more than 3/4 of the column is empty)
- `embarked` — only 2 missing

---

## Why I dropped those 6 columns

**deck** — 688 out of 891 values blank. Filling 688 blanks with an average means I'm basically inventing most of the column. That's not filling, that's making things up. Dropped.

**alive** — this column says "yes" or "no" for whether the person survived. That's literally the answer we're trying to predict. If I kept this during training the model would just look at it and score 100%, which is completely meaningless.

**class and embark_town** — contain the same information as `pclass` and `embarked` just in a slightly different format. Having both is redundant.

**who and adult_male** — these tell you man/woman/child, which is already there in `sex` and `age`. Repeating it doesn't help.

Side note — always worth checking before training if any column is secretly the same as another, or is already the answer.

---

## Dealing with missing values

**age (177 missing)** — filled with average age (~29.7) using `df['age'].fillna(df['age'].mean())`

Fine to do here because 177 is about 20% and ages average out sensibly. Wouldn't do this for deck where 77% is blank — that's too much to fill in, you'd be guessing most of the column.

**embarked (2 missing)** — dropped those 2 rows. 891 → 889. Losing 2 rows is nothing.

---

## Converting text to numbers

`sex` — female=0, male=1 using LabelEncoder
`embarked` — Cherbourg=0, Queenstown=1, Southampton=2 using LabelEncoder
`alone` — was True/False, ran `df.astype(int)` to turn everything into 0s and 1s

Went with label encoding and not one-hot because Logistic Regression doesn't assume that Southampton (=2) means something bigger or better than Cherbourg (=0). They're just different labels. If I used a decision tree later I'd probably switch to one-hot to be safe since those can be more sensitive to number values.

---

## Why can't we just use Linear Regression

Linear Regression can output any number. Ask it if someone survived and it might say 1.7 or -0.4. Those mean nothing when the answer has to be 0 or 1.

Logistic Regression runs the result through the **sigmoid formula**. No matter what number you put in, what comes out is always between 0 and 1:

$$P = \frac{1}{1 + e^{-z}}$$

Where:

- $z$ = weighted sum of all input columns (same as Linear Regression): $z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$
- $e$ = 2.718 (just a fixed maths number)
- $P$ = probability the person survived, always between 0 and 1

So if $P$ comes out as 0.73 → 73% chance they survived → predict 1 (survived). If it's 0.3 → predict 0.

The 0.5 cutoff point isn't fixed. For something like a cancer test, you'd drop it to 0.3 or lower — better to flag more people and check them than miss someone who's actually sick.

---

## Confusion matrix — how to read it

With a number prediction R² told you how close the guesses were. With a yes/no prediction, the only question is: did you pick the right box?

Confusion matrix is a 2x2 table:

|           | You said 0                          | You said 1                            |
| --------- | ----------------------------------- | ------------------------------------- |
| **Was 0** | True Negative — got it right ✓      | False Positive — wrong, false alarm ✗ |
| **Was 1** | False Negative — wrong, missed it ✗ | True Positive — got it right ✓        |

Our result:

```
[[90  19]
 [16  53]]
```

- TN = 90 — didn't survive, model said no — right
- FP = 19 — didn't survive, model said yes — wrong
- FN = 16 — survived, model said no — wrong, missed them
- TP = 53 — survived, model said yes — right

---

## The four scores from the confusion matrix

**Accuracy** — of all 178 predictions, how many were right

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{53 + 90}{178} = 0.803$$

143 out of 178 correct. 80.3%.

**Precision** — when we said someone survived, how often was that right

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{53}{53 + 19} = 0.74$$

74% of our "survived" predictions were correct.

**Recall** — of everyone who actually survived, how many did we catch

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{53}{53 + 16} = 0.77$$

We caught 77% of the real survivors. Missed 16.

**F1** — one score that covers both

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.74 \times 0.77}{0.74 + 0.77} = 0.75$$

When to care about which one:

- Precision — when a false alarm is a big problem (spam filter shouldn't delete real emails)
- Recall — when missing a case is a big problem (medical test shouldn't miss sick patients)
- F1 — when you want one number that balances both

---

## Why accuracy on its own can be useless

Say only 10 people out of 1000 have a disease. A model that just always says "no disease" would score 99% accuracy. Sounds amazing. Actually does nothing useful.

Our test set was 109 non-survivors and 69 survivors — not as extreme but same idea. Always look at Precision, Recall and F1 alongside accuracy when one answer comes up more than the other.

---

## Final scores

|           | Non-survivor | Survivor |
| --------- | ------------ | -------- |
| Precision | 0.85         | 0.74     |
| Recall    | 0.83         | 0.77     |
| F1        | 0.84         | 0.75     |

Overall: 80.3%

Better at non-survivors because the training data had more of them. Model practiced on that side more.

---

## What I'd try next

- Drop `fare`, `sibsp`, `parch` and see if score drops — suspect `sex`, `pclass` and `age` are doing most of the actual work
- Random Forest — can catch things like "female AND first class survived more" together, which a straight line can't really do
- `class_weight='balanced'` in LogisticRegression — stops the model quietly favouring non-survivors just because there are more of them in the training data

---

_Part of the Algorithms/Supervised/Classification series._
