# Notes — things I want to remember from this

---

## First look at the data

Before doing anything I just ran a few basic commands to understand the dataset. This is something I should always do before jumping into any model building.

**df.head()**

This just prints the first 5 rows so you can see what the data actually looks like. Not the column names — the actual values. Here I could see things like Fiesta in the model column, 2017 in year, 12000 in price, Manual in transmission and so on. Immediately tells you whether the data makes sense or looks messy. In this case it looked clean and readable.

**df.shape**

Gives you two numbers — rows and columns. Got (17966, 9) which means 17,966 cars and 9 columns. Good amount of data. Having too few rows would mean the model can't really learn properly, so 18k is fine.

**df.info()**

This one tells you two important things at once — what type of data each column holds, and whether any column has missing values.

Type of data matters because text columns and number columns need to be handled differently. Here 3 columns were text (model, transmission, fuelType) and 6 were numbers. The text ones would need to be converted before the model could use them.

Missing values — if a column has 17966 rows but only 16000 have data, 1966 values are missing and you have to either fill them in or drop those rows. Luckily here every column had all 17966 values filled. Nothing missing, no work needed on that front.

**df.describe()**

This prints basic stats for all the number columns — minimum value, maximum, average, and where the middle 50% of the data sits. You use this to spot anything strange.

A few things I noticed:

- Average selling price was around £12,279 — so mostly normal second hand cars, not exotic stuff
- The year column had a maximum of 2060 — obvious mistake, either a typo in the data or someone listing a car pre-emptively. Didn't clean it up this time but it could theoretically throw the model off slightly
- Cheapest car was £495 and most expensive was £54,995 — big range
- One car had 201.8 mpg which is physically impossible for a regular car. Probably another data entry error

These things don't necessarily stop you from training a model but it's good to know they exist before drawing any conclusions.

**df.isnull().sum()**

Checks how many missing values each column has. Got zero for every single column. That's the ideal situation — means no rows need to be dropped and no values need to be filled in with guesses.

---

## What the charts showed

After the basic checks I plotted a few charts to visually understand which columns actually affect the price. The idea is — if two things move together (one goes up when the other goes up, or one goes down when the other goes up), they're probably related and the column is worth keeping.

**Price histogram**

Just plotted all the prices to see how they're spread out. Most cars are priced between 8k and 15k. There are some expensive ones toward 50k+ but very few. This kind of shape (lot of cars in the middle, fewer at the edges) is pretty normal for a used car market.

**Heatmap of number columns**

A heatmap shows all the number columns and how much each one is related to every other one. The darker or more intense the colour in a box, the stronger the relationship. You're mainly looking at the "price" row/column to see which other columns move with it.

year had a strong positive link with price — newer car, higher price. Makes complete sense.
mileage had a strong negative link — more miles driven, lower price. Also makes sense.
engine size had some positive link — bigger engine tends to be pricier.
tax and mpg had weaker links to price. Included them anyway since they did have some relation.

**Mileage vs price scatter plot**

Plotted every car as a dot on a graph with mileage on one axis and price on the other. The dots go from top-left (low mileage, high price) to bottom-right (high mileage, low price). Was obvious to guess this would be the case but it's always better to confirm it in the actual data rather than assume.

**Engine size vs price box plot**

Grouped the cars by engine size and showed the price spread for each group. 1.0L cars were noticeably cheaper, 2.0L and above were in a higher range. Makes sense since bigger engines usually mean more powerful or more premium cars.

**Fuel type vs price box plot**

Grouped by petrol, diesel and hybrid. Diesel and hybrid cars were generally priced a bit higher than petrol ones. Probably because diesel cars are usually bigger and hybrid cars have extra technology in them. Hadn't thought about this before but the chart made it clear.

**Transmission vs price box plot**

Grouped by manual, automatic and semi-auto. Automatics were slightly more expensive on average. Not a huge difference but it's there. Probably because automatic gearboxes cost more to make.

**Model vs price box plot**

This was the most interesting one. The car model had the biggest impact on price out of all the text columns. Mustang was way at the top, smaller city cars like Ka were at the bottom. Same brand (Ford) but the price varies massively depending on the exact model. This made it very clear that the model column needs to be properly handled — you can't just ignore it or treat all models as equal.

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
