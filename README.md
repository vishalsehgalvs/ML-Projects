# ML Projects

A learning portfolio built while going through machine learning from scratch — theory notes, algorithm exercises, and real applied projects all in one place.

## Structure

```
ML-Projects/
├── Notes/          — personal study notes (AI vs ML, supervised vs unsupervised, key concepts)
├── Algorithms/     — algorithm exercises using the same Titanic dataset so results are comparable
│   ├── Supervised/
│   │   ├── Regression/         — Linear Regression (Ford used car price prediction)
│   │   └── Classification/     — 5 classifiers compared head to head on Titanic survival
│   │       ├── Logistic_Regression/
│   │       ├── KNN/
│   │       ├── Naive_Bayes/
│   │       ├── Decision_Tree/
│   │       └── SVM/
│   └── Unsupervised/       — Clustering, Dimensionality Reduction (🔜 coming)
└── Projects/       — real-world applied work on actual datasets
    ├── Insurance_Premium_Prediction/    — EDA + Linear Regression on Kaggle insurance premiums
    ├── Heart_Disease_EDA/               — EDA, cleaning, encoding and scaling on heart disease data
    └── Heart_Disease_Model_Comparison/  — compare all 5 classifiers on heart disease data + app
        └── Heart_Disease_Prediction_App/ — Streamlit app for live heart disease risk prediction
```

## Algorithms — Titanic Survival (Classification)

All five classifiers trained on the same dataset so you can compare fairly:

| Algorithm           | Accuracy  | Notes                                 |
| ------------------- | --------- | ------------------------------------- |
| Logistic Regression | 80.3%     | Good baseline, linear boundary        |
| KNN (K=5)           | 78.1%     | Distance-based, needs scaling         |
| Naive Bayes         | 77.5%     | Fast, assumes feature independence    |
| Decision Tree       | 77.0%     | Interpretable, prone to overfitting   |
| SVM (RBF)           | **82.6%** | Best on Titanic — max margin boundary |

## Projects

| Project                                                                                                | What it does                                                                                   | Status      |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ----------- |
| [Insurance_Premium_Prediction](./Projects/Insurance_Premium_Prediction)                                | EDA, data cleaning, feature selection and Linear Regression on Kaggle insurance premiums       | ✅ Complete |
| [Heart_Disease_EDA](./Projects/Heart_Disease_EDA)                                                      | Exploratory data analysis, encoding and scaling on a heart disease dataset                     | ✅ Complete |
| [Heart_Disease_Model_Comparison](./Projects/Heart_Disease_Model_Comparison)                            | Train and compare all 5 classifiers on heart disease data — Logistic Regression wins at 86.96% | ✅ Complete |
| [Heart_Disease_Prediction_App](./Projects/Heart_Disease_Model_Comparison/Heart_Disease_Prediction_App) | Streamlit web app — fill in health details, get a risk prediction                              | ✅ Complete |

## Notes

General study notes on the theory side:

| File                                                                   | Topic                                                             |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [ai_vs_ml_vs_dl.md](./Notes/ai_vs_ml_vs_dl.md)                         | What's the difference between AI, ML and Deep Learning            |
| [supervised_vs_unsupervised.md](./Notes/supervised_vs_unsupervised.md) | Supervised vs unsupervised learning explained                     |
| [ml_concepts.md](./Notes/ml_concepts.md)                               | Key ML concepts — overfitting, bias-variance, feature scaling etc |

---

_More algorithms and projects added as I learn._
