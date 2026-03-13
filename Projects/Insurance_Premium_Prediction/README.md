# Insurance ML Project

This is my first ML project. I picked up a Kaggle insurance dataset and worked through the full process — loading the data, cleaning it, exploring it with charts, and figuring out which columns actually matter for predicting insurance charges.

## Folder Structure

- **insurance_ml_project.py**: Main Python script for data analysis and ML workflow.
- **insurance.csv**: Dataset containing insurance records (age, sex, bmi, children, smoker, region, charges).
- **images/**: All visualisation plots generated during analysis:
  - `age boxplot.png`, `age plot distribution.png`
  - `bmi boxplot.png`, `bmi plot distribution.png`
  - `charges boxplot.png`, `charges plot distribution.png`
  - `children boxplot.png`, `children count plot.png`, `children plot distribution.png`
  - `data heatmap.png`, `gender plot.png`, `smoker plot.png`

## What I Did (Step by Step)

1. **Loaded the data**
   - Read `insurance.csv` into pandas and checked what it looks like — column types, basic stats, any missing values.
   - Set `pd.set_option('display.max_columns', None)` and `pd.set_option('display.width', None)` right after imports — otherwise pandas cuts off columns in the terminal and you see `...` instead of the actual values.

2. **Explored the data visually (EDA)**
   - Plotted distributions, boxplots, count plots, and a heatmap to understand the data before touching it.

3. **Cleaned and prepared the data**
   - Dropped duplicate rows.
   - Converted text columns (`sex`, `smoker`, `region`) into numbers so the model can read them.
   - Created a new BMI category column (underweight / normal / overweight / obese).
   - Scaled numeric columns so they're all on the same scale.

4. **Picked the most important features**
   - Used Pearson Correlation to see which columns are linked to charges.
   - Used Chi-Square tests to check if categorical columns are statistically relevant.
   - Dropped the ones that didn't matter.

5. **Split the data and trained a model**
   - Kept 80% of the rows to train the model on and held back 20% to test it at the end.
   - The 20% test rows are something the model has never seen — so the score on them is honest.
   - Trained a Linear Regression model on the training rows.
   - Checked the score on the test rows to see how well it did.

## Visualizations

### Age Analysis

![Age Distribution](images/age%20plot%20distribution.png)
![Age Boxplot](images/age%20boxplot.png)

### BMI Analysis

![BMI Distribution](images/bmi%20plot%20distribution.png)
![BMI Boxplot](images/bmi%20boxplot.png)

### Children Analysis

![Children Distribution](images/children%20plot%20distribution.png)
![Children Count Plot](images/children%20count%20plot.png)
![Children Boxplot](images/children%20boxplot'.png)

### Charges Analysis

![Charges Distribution](images/charges%20plot%20distribution.png)
![Charges Boxplot](images/charges%20boxplot.png)

### Categorical Features

![Gender Plot](images/gender%20plot.png)
![Smoker Plot](images/smoker%20plot.png)

### Correlation Heatmap

![Data Heatmap](images/data%20heatmap.png)

## Key Libraries Used

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- scipy

## Dataset Example

| age | sex    | bmi   | children | smoker | region    | charges   |
| --- | ------ | ----- | -------- | ------ | --------- | --------- |
| 19  | female | 27.9  | 0        | yes    | southwest | 16884.924 |
| 18  | male   | 33.77 | 1        | no     | southeast | 1725.5523 |
| ... | ...    | ...   | ...      | ...    | ...       | ...       |

## How to Run

1. Run `insurance_ml_project.py` — it will print results at each step.
2. To see the plots, uncomment the plot lines in the EDA section.
3. Feel free to use this as a reference — that's what it's here for.

## Key Findings

| Feature              | Pearson Correlation | Significance                   |
| -------------------- | ------------------- | ------------------------------ |
| `is_smoker`          | 0.787               | Strongest predictor of charges |
| `age`                | 0.298               | Moderate positive correlation  |
| `bmi_category_obese` | 0.200               | Higher BMI → higher charges    |
| `bmi`                | 0.196               | Moderate positive correlation  |
| `is_female`          | -0.058              | Weak negative correlation      |

**Chi-Square Test results (p < 0.05 → keep feature):**

| Feature                   | Decision |
| ------------------------- | -------- |
| `is_smoker`               | ✅ Keep  |
| `region_southeast`        | ✅ Keep  |
| `is_female`               | ✅ Keep  |
| `bmi_category_obese`      | ✅ Keep  |
| `region_southwest`        | ❌ Drop  |
| `bmi_category_overweight` | ❌ Drop  |
| `bmi_category_normal`     | ❌ Drop  |
| `region_northwest`        | ❌ Drop  |

## Final Selected Features

After both Pearson correlation and Chi-square analysis, the following 8 features were selected for model training:

```
age | is_female | bmi | children | is_smoker | region_southeast | bmi_category_obese | charges
```

## Model Training and Results

Went with Linear Regression first since it's the simplest starting point. You give it the input columns and the charges column, it figures out a formula that connects them, and then uses that formula to guess charges for rows it hasn't seen.

**Train / Test split**

Split the data 80/20. 80% went into training, 20% got set aside and weren't touched until the very end. The reason you do this — if you test on the same data you trained on, the model has already seen those rows and knows the answers. That's not a real test. The 20% it's never seen is the honest check.

**Results**

| Metric      | Score  |
| ----------- | ------ |
| R²          | 0.8041 |
| Adjusted R² | 0.7988 |

R² of 0.80 means the model explained about 80% of why some people pay more than others. For a first attempt with the most basic model, that's decent.

Adjusted R² being almost the same (0.7988 vs 0.8041) means the 8 columns we kept are actually doing useful work. If some columns were just dead weight, adjusted R² would've dropped noticeably below plain R².

**Overfitting and Underfitting**

Overfitting — model scores great on training rows but badly on test rows. It basically crammed the training data instead of learning the actual pattern. Like a student who memorised only last year's JEE paper and blanks on anything worded slightly differently in the real exam.

Underfitting — bad on both. Model just didn't learn anything useful.

Scored around 80% on both here so neither was an issue.

**What to try next**

Ridge and Lasso — same as Linear Regression but with a small penalty that stops the model from leaning too hard on just one or two columns. Worth trying to see if the score improves.

Cross validation — do the 80/20 split 5 times with different rows each time and average the scores. Less dependent on luck about which rows ended up in the test set.

## What I Learned

- Always check the data first before touching anything — head, info, describe, nulls
- Duplicate rows exist in real datasets and need to be dropped
- Text columns have to become numbers before a model can use them
- Sometimes you create a new column from an existing one (did this with bmi → bmi category)
- Scaling matters — without it big numbers like mileage or charges dominate just because of their size
- Pearson Correlation is how you check which number columns are linked to what you want to predict
- Chi-Square is the same idea but for yes/no and category columns
- Never test on the same rows you trained on — that's not a real test
- R² tells you how much of the answer the model got right, Adjusted R² tells you if the columns you used are actually earning their place
- 80% on a simple first model is a good starting point

---

_My first ML project. Dataset from Kaggle._
