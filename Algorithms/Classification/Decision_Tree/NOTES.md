# Notes — Decision Tree

Writing this down while going through the algorithm so it actually sticks.

---

## How it works

A decision tree is basically a flowchart the computer builds for you.

Think of the game 20 Questions. Someone thinks of a thing, and you ask yes/no questions to narrow it down. "Is it alive?" — "Is it bigger than a car?" — "Does it have four legs?" Each question cuts the possibilities in half. A decision tree does the same thing with your data.

Applied to Titanic: the model looks at 711 training passengers and figures out which questions to ask — and in what order — to best separate survivors from non-survivors. Something like:

- Is the passenger female?
  - Yes → probably survived
  - No → is their class 1st?
    - Yes → maybe survived
    - No → probably didn't

The model doesn't ask those specific questions — it figures out the best ones from the data automatically. But that's the shape of what it builds.

---

## How it decides which question to ask first

This is the part that confused me at first. The tree has to choose which column to split on, and at what value. It does this using something called **Gini Impurity**.

Gini measures how mixed a group is. If you have a pile of 100 passengers and 99 of them survived — that's a clean pile. Low impurity. If 50 survived and 50 didn't — totally mixed. High impurity.

$$\text{Gini} = 1 - \sum p_i^2$$

Where $p_i$ is the fraction of each class in the group. A pure group (all one class) gives Gini = 0. An even 50/50 split gives Gini = 0.5.

At every split, the algorithm tries every possible question — every column at every possible cutoff value — and picks whichever one reduces impurity the most. Then it does this again on each branch, recursively, until it either runs out of things to split on or hits a stopping rule.

There's also **Entropy** which measures the same thing differently, borrowing from information theory. Both work. sklearn defaults to Gini.

---

## What "depth" of a tree means

The depth is how many levels of questions the tree goes through before making a final call.

- Depth 1 — one question, two buckets. Very rough.
- Depth 3 — three levels of questions. More nuanced.
- No limit — the tree keeps going until every leaf has only one class.

A tree with no depth limit will memorise the training data. It'll ask such specific questions that it perfectly classifies every training passenger — but those questions are so tailored to the training data that they fall apart on new passengers. This is called **overfitting**.

Think of it this way: if I study only past exam papers and memorise every answer without understanding the subject, I'll ace the practice papers but fail a new one with slightly different wording. Same idea.

In this code, `random_state=42` is set but no `max_depth` is given — so the tree grows until all leaves are pure. Worth experimenting with max_depth later to see if limiting it improves the test score.

---

## Does scaling matter for decision trees?

Strictly speaking, no. Decision trees split on threshold values — "is age > 30?" — and whether you scale the data or not doesn't change where the best threshold is relative to the data distribution.

Logistic Regression and Decision Trees don't actually need scaling. KNN does because it measures distance.

In this code the scaler is applied anyway before fitting the decision tree. The accuracy isn't wrong because of it — the tree will still find the right splits — but it's not something the algorithm requires. Scaling won't hurt it, it just doesn't help either.

---

## Results

```
Accuracy: 76.97%

Confusion Matrix:
[[88  21]
 [20  49]]
```

Out of 178 test passengers:

- 88 — said died, actually died. correct.
- 49 — said survived, actually survived. correct.
- 21 — said survived, actually died. false alarm.
- 20 — said died, actually survived. missed them.

```
Classification Report:
              precision    recall    f1
   Died (0)     0.81       0.81     0.81
Survived (1)   0.70       0.71     0.71
```

---

## All four algorithms compared now

|                 | Logistic Regression | KNN K=5 | Naive Bayes | Decision Tree |
| --------------- | ------------------- | ------- | ----------- | ------------- |
| Accuracy        | **80.3%**           | 78.1%   | 77.5%       | 77.0%         |
| Died recall     | 0.83                | 0.83    | 0.77        | 0.81          |
| Survived recall | 0.77                | 0.71    | **0.78**    | 0.71          |

Decision Tree comes in last here. On this dataset with this setup, it's not beating the others. But the gap is still within 3–4% — and this tree hasn't been tuned at all. Adding `max_depth` or `min_samples_split` would likely close that gap.

The more interesting thing is how different the algorithm is in concept from everything before it, yet the numbers land in roughly the same place. Cleaning the data well continues to matter more than the algorithm choice.

---

## What's genuinely good about decision trees

The big one: you can actually see what it learned. With Logistic Regression you get a bunch of coefficients. With KNN there's no "model" to look at. With a decision tree you can print the literal flowchart — "if sex <= 0.5 and pclass <= 1.5 then survived". A non-technical person could follow it.

That interpretability is why decision trees show up a lot in places like medical diagnosis, loan approval, and fraud detection — situations where you need to explain the decision, not just make it.

They're also not bothered by features being on different scales, they handle non-linear relationships naturally, and they work for both classification and regression.

The weakness is overfitting. An unconstrained tree memorises the training data. The fix to that is Random Forests — which builds hundreds of trees, each on a random subset of the data, and averages their predictions. But that's for later.

---

## Things to remember

- Decision tree = flowchart of yes/no questions. Each split tries to make the groups as pure as possible.
- Gini impurity measures how mixed a group is. Lower is better.
- No max_depth = the tree memorises training data. Use max_depth to control this.
- Scaling isn't needed for decision trees — splits are based on thresholds, not distances.
- The big advantage over other algorithms is that you can actually read what the model decided and why.

---

_Part of the Algorithms/Classification series._
