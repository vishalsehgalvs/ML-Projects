# Notes — things I want to remember from this

---

## One-hot vs label encoding — actually a big deal

I didn't expect the encoding choice to matter this much. 84% vs 73% on the same data, same model, same everything. Just the encoding changed.

The reason label encoding did badly: when you tell the model Fiesta=5 and Focus=6, it treats that like Focus is somehow 1 unit more than Fiesta. Linear Regression picks up on that and acts like there's a real relationship there. There isn't — they're just two different cars, neither is "more" than the other.

One-hot handles this properly — each model gets its own column, completely separate, no implied ranking.

I'll always default to one-hot now unless the categories have an actual order (like T-shirt sizes, survey ratings, anything where small < medium < large makes sense).

---

## Adjusted R² — why it exists

Basic R² will never go down when you add more columns, even if those columns are completely useless. So you could technically throw in a random column and R² would either stay the same or tick up slightly — even though the column added nothing.

Adjusted R² actually penalises you for adding dead weight columns. If it drops a lot compared to plain R², some of your columns are freeloading. If both are close to each other, the columns you kept are genuinely doing something.

Here both were around 84.5% so the columns were fine.

---

## drop_first=True in get_dummies — what's that about

If a car is not Fiesta and not Focus and not Kuga and not Galaxy etc., then it must be the last one — so you don't need a column for it. That last column is already captured by all the others being 0.

If you don't drop it, you end up with duplicate information which can confuse the model. `drop_first=True` handles this automatically.

---

## Why I scaled the numbers

Mileage can be 80,000. Year is around 2016. Tax might be 150. Engine size is 1.2.

Without scaling, the model would naturally pay more attention to mileage just because the number is huge compared to the others. StandardScaler brings everything to the same ballpark so the model judges each column on its actual importance, not the size of the number.

Only scaled the proper number columns — the encoded ones are already 0 or 1 so they don't need it.

---

## Spotted something odd in the data

The year column had a max of 2060. Either someone listed a car from the future or it's a typo. Didn't fix it this time but worth checking before using this dataset for anything serious.

---

## What to try next

- Ridge and Lasso — same as linear regression but with a knob that stops it from leaning too hard on any one column. Useful if the model starts overfitting.
- Look at which columns had the most weight after training — curious how much is_smoker-equivalent columns matter here compared to year or mileage.
- Cross validation — do the 80/20 split multiple times and average instead of relying on one split.
