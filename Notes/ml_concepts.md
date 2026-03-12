# ML Concepts

Things I kept having to look up. Writing them here so I stop googling the same stuff.

---

## Features and Labels

Features = input columns. All the things you know about each row — age, salary, cholesterol, city, whatever. Label = the one column you're trying to predict — price, yes/no for disease, fraud/not fraud etc. That's it. Just two words for inputs and output.

---

## Train / Test Split

You hold back some data before training and only look at it at the very end to check how the model does on stuff it hasn't seen. If you tested it on the same data you trained it on, it'd just memorise the answers — same as a student getting the exact same JEE question paper to practice on that later shows up in the actual exam. Gets everything right, learnt nothing.

Usual split is 80/20. 80% to train, 20% kept aside for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## Overfitting and Underfitting

Overfitting — model crammed the training data, didn't actually learn the pattern. Scores very high on training, falls apart on the test set. Classic example: student who mugged every CBSE past paper line by line, then got a twisted question in the board exam and blanked.

Underfitting — opposite problem. Model is too simple, didn't pick up the pattern even on training data. Does badly on both.

You want to be somewhere in the middle. If training score and test score are both decent and reasonably close to each other, that's a good sign.

---

## Bias and Variance

Just fancy words for the same problem. High bias = underfitting, model is too simple. High variance = overfitting, model is too tightly fitted to the training data. Making the model more flexible helps one but usually hurts the other. You're always trying to find the balance.

---

## Is the model actually good

Depends on what you're predicting.

For numbers (like predicting LIC premium or rent in Hyderabad): MAE is the average difference between what the model predicted and the actual value — easy to understand, same unit as what you're predicting. RMSE is similar but punishes bigger mistakes more. R² tells you how much of the variation the model actually explained — 1.0 is perfect, 0 means it's no better than just guessing the average every time.

For categories (like fraud yes/no, disease yes/no): Accuracy sounds good — what percentage did it get right — but it can be very misleading. If only 2% of UPI transactions are fraud, a model that just always says "not fraud" gets 98% accuracy and is completely useless. That's when you look at Precision and Recall. Recall is especially important for medical stuff — it tells you how many of the actual sick patients the model actually caught. Missing a real case is far worse than a false alarm. F1 score combines both into one number.

---

## Cross Validation

Instead of trusting one train/test split (which might just be a lucky or unlucky split), you run multiple splits and average the results. The standard way is 5-fold — cut data into 5 chunks, rotate which chunk is the test set, run 5 times, take the average score. More reliable than a single split.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())
```

---

## Scaling

Age goes up to maybe 80. Cholesterol can be 300+. Annual salary in rupees could be in the lakhs. Without scaling, the model might give more weight to bigger numbers just because they're bigger — not because they actually matter more. Scaling brings everything to a similar range so no column bullies the others.

I use StandardScaler by default — shifts each column so the average is 0. MinMaxScaler squashes everything between 0 and 1, useful when an algorithm specifically needs that range.

---

## Hyperparameters

Settings you choose before training that the model doesn't figure out itself. Like how many trees in a Random Forest, or how deep each tree is allowed to grow. You have to experiment with these to get better results. GridSearchCV automates the trying of combinations if you want to do it properly.

---

## My usual process

Load → look at it → clean it → encode/scale → split → train → check scores → adjust if needed.
