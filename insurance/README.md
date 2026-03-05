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

## What I Learned

- How to load and perform EDA on a real-world dataset
- Handling duplicate rows and verifying data quality
- Label encoding and one-hot encoding for categorical variables
- Creating new features from existing ones (BMI categories)
- Feature scaling with `StandardScaler`
- Using **Pearson Correlation** to measure linear relationships with the target
- Using **Chi-Square Tests** to assess independence of categorical features
- How to select a final, clean feature set ready for model training

---

_My first ML project. Dataset from Kaggle. Still learning — model training comes next._
