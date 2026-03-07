# Supervised vs Unsupervised Learning

Took me a bit to properly get this. But it's actually quite simple once you stop overthinking it.

---

## Supervised

You have data and every row already has the answer attached to it. The model learns from all those answered examples and then tries to predict the answer for new rows it hasn't seen.

I used this in my heart project — 918 patient records, each one already marked yes or no for heart disease. The model learned what combinations of age, cholesterol, chest pain type etc. tend to mean yes or no. Then you hand it a new patient's details and it makes a guess.

Two types depending on what you're predicting. If it's a number — like flat rent in Bangalore or someone's salary — that's Regression. If it's a category — like yes/no, fraud/not fraud, spam/not spam — that's Classification.

Algorithms I've come across so far: Linear Regression (predicts numbers, fits a straight line), Logistic Regression (confusing name but it's actually for categories, not numbers), Decision Tree (like a flowchart of yes/no questions), Random Forest (many decision trees voting together — more reliable), SVM (finds the clearest possible boundary line between two groups).

---

## Unsupervised

No labels at all. You just throw the raw data at it and it tries to find patterns by itself.

Think of a Flipkart data team with crores of customer purchase records but no one's labelled anything. Unsupervised learning might come back and say — look, there are basically 4 types of buyers here: bargain hunters, premium buyers, festive-sale-only people, and impulse buyers. Nobody told it to find those groups. It just found them on its own.

Two main uses: clustering (grouping similar rows — K-Means is the most common one) and reducing columns (if you have 50 columns and want to shrink that before training, PCA handles it).

---

## Quick way to decide which one to use

Do you have labels in your data? Yes → supervised. No → unsupervised. That's genuinely the whole decision.

---

## Reinforcement Learning

Just keeping a note of this because it comes up a lot. Not using it myself right now.

No labelled data here either. The model just tries things, gets told how good or bad each try was, and slowly figures out what works. Think of how a kid learns to play carrom — nobody gives them a manual, they just keep playing, see what shots work, and get better. Same idea. Used in game bots, robotics, self-driving cars. Not relevant for normal row-and-column datasets like what I'm working with.
