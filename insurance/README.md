# Insurance ML Project

A beginner-friendly machine learning project for insurance data analysis and feature selection. This project demonstrates data cleaning, preprocessing, visualization, and statistical analysis using Python.

## Folder Structure

- **insurance_ml_project.py**: Main Python script for data analysis and ML workflow.
- **insurance.csv**: Dataset containing insurance records (age, sex, bmi, children, smoker, region, charges).
- **images/**: All visualisation plots generated during analysis:
  - `age boxplot.png`, `age plot distribution.png`
  - `bmi boxplot.png`, `bmi plot distribution.png`
  - `charges boxplot.png`, `charges plot distribution.png`
  - `children boxplot.png`, `children count plot.png`, `children plot distribution.png`
  - `data heatmap.png`, `gender plot.png`, `smoker plot.png`

## Workflow Overview

1. **Data Loading**
   - Loads `insurance.csv` using pandas.
   - Displays basic info, summary statistics, and checks for missing values.

2. **Exploratory Data Analysis (EDA)**
   - Visualizes distributions and relationships for numeric features (age, bmi, children, charges).
   - Generates boxplots, histograms, count plots, and heatmaps (see images).

3. **Data Cleaning & Preprocessing**
   - Removes duplicates.
   - Encodes categorical variables (`sex`, `smoker`, `region`) to numeric.
   - Creates new features (e.g., BMI categories).
   - Scales numeric features using `StandardScaler`.

4. **Feature Selection & Statistical Analysis**
   - Calculates Pearson correlation for selected features vs. charges.
   - Performs Chi-square tests for categorical features.
   - Selects final features based on statistical significance.

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

## How to Use

1. Run `insurance_ml_project.py` to execute the analysis.
2. Review generated plots in the folder for insights.
3. Use the code as a reference for basic ML workflow and feature engineering.

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

_This project is for learning and reference purposes. Dataset source: Kaggle._
