# Model Training — Insurance Charges

All the messy prep work is done. This is the part where we actually build the model and see if it's any good.

---

## Which model

I went with Linear Regression to start. It's the simplest option for predicting a number like insurance charges — basically draws a best-fit line through the data and uses that to guess charges for new people. Not fancy but a solid first step before trying anything more complicated.

---

## Columns going into the model

Not all columns made it. After running the correlation and chi-square tests earlier, only these 7 were worth keeping:

| Column             | Meaning                                          |
| ------------------ | ------------------------------------------------ |
| age                | how old the person is                            |
| is_female          | 1 = female, 0 = male                             |
| bmi                | body weight relative to height                   |
| children           | how many kids they have                          |
| is_smoker          | smoker or not — this one matters the most by far |
| region_southeast   | whether they're from the southeast               |
| bmi_category_obese | whether their BMI puts them in the obese range   |

`charges` is what we're trying to predict — that got separated out as the target.

---

## Splitting the data

Before training, I split the data — 80% for training, 20% kept aside. The model only ever sees the 80% while learning. The 20% is used at the very end to check if it actually works on data it hasn't seen before. Without this split you'd just be testing it on the same questions it practised from.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

`random_state=42` just means the split is the same every time you run it — otherwise you'd get slightly different rows each time.

---

## How it did

| Score       | Value  |
| ----------- | ------ |
| R²          | 0.8041 |
| Adjusted R² | 0.7988 |

R² of 80% basically means — out of all the reasons why charges vary between people, the model is explaining about 80% of them. That's a decent result for a straight line through 7 columns.

Adjusted R² is just a stricter version of R² that goes down if you add useless columns. The fact that both numbers are almost the same tells me the 7 features I kept are actually pulling their weight.

---

## Overfitting and underfitting

I didn't do a full check here but something to keep in mind going forward.

If the model scores much higher on training data than test data, it's overfit — it basically crammed the training rows and can't handle anything different. Like a student who mugs last year's exact JEE paper and then gets a slightly twisted version of the same question and blanks.

If both scores are low, it's underfit — the model is too simple, not finding any real pattern.

Both scores sitting close at 80% is fine for now.

---

## What I want to try next

Linear Regression assumes everything is a straight-line relationship which isn't always true. A few things worth trying:

- Ridge and Lasso regression — same idea as linear but with a built-in control that stops the model from over-relying on any one column. Lasso goes a step further and can drop weak columns on its own.
- Cross-validation — instead of one lucky or unlucky 80/20 split, run it 5 times with different splits and average the score. More reliable.
- Check which columns are actually driving the predictions — would be interesting to see how much weight the model is putting on is_smoker vs age.
