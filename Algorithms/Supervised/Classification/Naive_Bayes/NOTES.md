# Notes — Naive Bayes

Same dataset as before. Same cleaning. Just swapping out the algorithm to see how Naive Bayes compares.

---

## The core idea

Naive Bayes doesn't find patterns the way other algorithms do. It doesn't draw a line through data or measure distances. It just calculates probabilities.

For every new passenger it looks at, it asks two questions:

- What's the probability this person survived, given everything I know about them?
- What's the probability they died?

Whichever probability is higher — that's the prediction. Simple.

The formula behind it is called Bayes' Theorem. Looks like this:

$$P(\text{survived} \mid \text{features}) = \frac{P(\text{features} \mid \text{survived}) \times P(\text{survived})}{P(\text{features})}$$

Breaking it down in plain words:

- P(survived | features) — what we want. Probability of surviving given their age, sex, class, etc.
- P(features | survived) — among people who actually survived, how often did we see these kinds of features? Model figures this out during training.
- P(survived) — what fraction of passengers survived overall? Easy to count.
- P(features) — just a normalising number so the probabilities add up properly.

You don't calculate this yourself. sklearn handles all of it. But it helps to know what it's doing.

---

## Why it's called "Naive"

The algorithm assumes all features are completely independent of each other — meaning knowing someone's age tells you nothing about their ticket class, and knowing their fare tells you nothing about their sex.

That's obviously not true. Rich passengers were more likely to be older. Women were more likely to survive regardless of class. These things are connected.

But the algorithm just ignores all that. It treats every column as if it lives in its own bubble. That's why it's called naive — it's making an assumption it knows is wrong.

Here's the strange thing though: it still works. The predictions are often surprisingly decent even when the independence assumption is completely off. The explanation is that even when the probability numbers are wrong in absolute terms, they're usually still pointing in the right direction. The survived pile comes out higher than the died pile when it should, and that's enough.

---

## Gaussian — what does that mean

There are different versions of Naive Bayes depending on what kind of data you're working with. We used GaussianNB.

- GaussianNB — for continuous numbers like age and fare. Assumes each feature follows a bell curve (normal distribution).
- MultinomialNB — for word counts. Classic use case is spam detection.
- BernoulliNB — for features that are strictly 0 or 1.

Our data is continuous (age, fare, pclass) so Gaussian is the right one. During training it just calculates the mean and standard deviation for each feature in each class — survived pile and died pile separately. That's the whole model.

---

## No scaling needed here

With KNN I had to scale everything because KNN measures distance, and a fare of 500 completely drowns out a pclass of 3 if you don't scale.

Naive Bayes doesn't measure distance at all. It builds a separate probability distribution per feature. So fare having bigger numbers than pclass doesn't matter — they each get their own distribution and probabilities are calculated independently for each one.

So the scaler is imported in the code but not actually used. That's intentional, not a mistake.

---

## Data cleaning — same as before

Started with 891 rows and 15 columns.

Dropped 6 columns:

- deck — 688 out of 891 blanks. Can't use it.
- alive — same as survived written as yes/no. Keeping it would be cheating.
- class — same as pclass. duplicate.
- embark_town — same as embarked. duplicate.
- who and adult_male — both just restate what sex and age already tell you.

Missing values:

- age had 177 blanks — filled with the mean (~29.7)
- embarked had 2 blanks — just dropped those rows. 891 becoming 889 isn't a loss worth worrying about.

Encoding:

- sex — female=0, male=1
- embarked — Cherbourg=0, Queenstown=1, Southampton=2
- alone was True/False — converted everything to int so it becomes 1/0

80/20 split. 711 rows for training, 178 held back for testing. Same random_state=42 as before.

---

## Results

```
Accuracy: 77.5%

Confusion Matrix:
[[84  25]
 [15  54]]
```

So out of 178 test passengers:

- 84 — said they died, they died. correct.
- 54 — said they survived, they survived. correct.
- 25 — said they survived, they actually died. false alarm.
- 15 — said they died, they actually survived. missed them.

```
Classification Report:
              precision    recall    f1
   Died (0)     0.85       0.77     0.81
Survived (1)   0.68       0.78     0.73
```

The recall for survivors (0.78) is actually the highest of all three algorithms — higher than Logistic Regression (0.77) and well above KNN (0.71). Naive Bayes misses fewer actual survivors than the others. It does have more false alarms (25 vs 19) but it's better at not letting a real survivor slip through as a predicted death.

---

## How all three compare

|                    | Logistic Regression | KNN K=5 | Naive Bayes |
| ------------------ | ------------------- | ------- | ----------- |
| Accuracy           | 80.3%               | 78.1%   | 77.5%       |
| Died precision     | 0.85                | 0.82    | 0.85        |
| Survived precision | 0.74                | 0.72    | 0.68        |
| Died recall        | 0.83                | 0.83    | 0.77        |
| Survived recall    | 0.77                | 0.71    | **0.78**    |

Logistic Regression still wins on accuracy. But the gap between all three is tiny — less than 3%. Which makes sense because all three are working with the exact same cleaned data. The cleaning probably mattered more than the algorithm choice here.

---

## When Naive Bayes is actually worth using

On structured tabular data like this Titanic set, it's not the go-to algorithm. Features in real datasets are usually correlated and the independence assumption hurts it.

Where it genuinely shines is text. Spam filtering. Sentiment analysis. Document classification. The reason is that in a bag-of-words approach, word frequencies are roughly independent enough for the assumption to hold up reasonably well, and the speed advantage matters when you're classifying thousands of emails in real time.

It's also surprisingly good when you have very little training data — it doesn't need thousands of rows to start making decent predictions.

---

## Quick takeaways

- Naive Bayes = probability-based, not distance-based. No scaling needed.
- GaussianNB is for continuous features. Training is just computing means and standard deviations.
- The independence assumption is almost always wrong, but results are usually still good.
- On this dataset it came 3rd on accuracy but 1st on survivor recall.
- All three classifiers were within 3% of each other — data prep matters more than the algorithm.

---

_Part of the Algorithms/Supervised/Classification series._
