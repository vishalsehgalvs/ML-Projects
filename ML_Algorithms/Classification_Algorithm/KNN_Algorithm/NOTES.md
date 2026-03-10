# Notes — KNN (K-Nearest Neighbours)

Writing this down before I start coding so I actually understand what I'm doing.

---

## How it works

KNN doesn't train a model the way Logistic Regression does. It just holds onto all the training data. When you throw a new row at it, it:

1. Measures how far that row is from every row in the training data
2. Picks the K closest ones
3. Checks what category those K rows belong to and picks the most common one

So if K=5 and 4 of the 5 nearest rows are "survived", it predicts survived.

---

## How distance is calculated

Default is straight-line distance between two points — same as Pythagoras:

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

When there are more than 2 columns, just keep adding more terms under the square root — one for each column.

There's another way called Manhattan distance — instead of squaring, just take the absolute difference for each column and add them up:

$$d = |x_2 - x_1| + |y_2 - y_1|$$

Think of it as walking on a city grid vs flying in a straight line.

---

## Why you have to scale the columns first

Say one column is age (values around 20–80) and another is salary (values around 20000–100000). When you calculate distance, salary is going to completely take over just because its numbers are so much bigger. Age barely factors in.

Scaling brings all columns to the same range so they all count equally. Either:

- StandardScaler — shifts values so the average is 0, most values fall between -3 and 3
- MinMaxScaler — squishes everything to 0–1

This step matters way more for KNN than it does for most other algorithms.

---

## Picking K

- K=1 — looks at only the single closest match. Too sensitive. One odd row in the data can mess up a prediction.
- K very high — ends up just predicting whatever the most common category is. Too lazy.
- Good starting point — try K=3, 5, 7. Odd numbers avoid ties.

Usually just run it for a range of K values, plot accuracy, and pick where it stops improving.

---

## What's good and bad about it

**Good:**

- Easy to understand — if you can explain "find the most similar ones and vote" you've explained the whole thing
- No training step — just saves the data
- Works for problems with more than 2 categories without any extra setup

**Bad:**

- Slow when predicting — has to check every single training row for every new prediction
- Uses a lot of memory — keeps the entire training set around
- Struggles when there are lots of columns — distances start meaning less when everything is far from everything
- Breaks if you forget to scale

---

## Checking results

Same tools as Logistic Regression — confusion matrix, accuracy, precision, recall, F1. Nothing new here.

---

_Will fill in the dataset-specific stuff once I pick one and run the code._
