# IBM HR Employee Attrition Modeling

An end-to-end Machine Learning pipeline evaluating classification algorithms and class-imbalance strategies on the IBM HR Analytics Attrition dataset.

## Project Overview
This repository contains a baseline workflow for predicting voluntary employee turnover using 1,470 tabular records. The target variable (`Attrition`) exhibits an ~84/16 class imbalance. 

The primary goal was to establish a leak-free modeling pipeline, evaluate tree vs. linear models, and tune decision thresholds on a holdout set to manage precision/recall trade-offs.

---

## Data Pipeline & Validation Strategy
To avoid data leakage, all feature transformations, scaling, and threshold tuning were strictly isolated to training and validation splits.

1. **Split Setup:** 70% Train / 15% Dev (Validation) / 15% Test.
2. **Preprocessing:** Numeric features scaled via `RobustScaler`. Categorical features encoded via One-Hot Encoding.
3. **Threshold Calibration:** Probability thresholds were evaluated on the Dev set to maximize the minority class F1-Score prior to final test evaluation.

---

## Key Results & Model Comparison

| Model Architecture | Resampling / Weights | Dev F1-Score | Test Recall (Class 1) | Test Precision (Class 1) | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost / XGBRF | None | ~0.42 | 40% | 45% | 0.42 |
| Logistic Regression | SMOTE | ~0.46 | 68% | 35% | 0.46 |
| **Logistic Regression** | **Balanced Weights** | **~0.59** | **58%** | **57%** | **0.58** |

### Final Test Performance (Logistic Regression @ 0.62 Threshold)
* **Overall Accuracy:** 86%
* **Attrition Class Recall:** 58% (21 / 36 leavers detected)
* **Attrition Class Precision:** 57%
* **Macro F1-Score:** 0.58

---

## Summary of Findings
* **Resampling Trade-offs:** SMOTE inflated training recall but produced noisy decision boundaries that degraded test precision. Cost-sensitive weighting produced more consistent results.
* **Tree Model Performance:** Tree ensembles overfit the small sample size ($N=1,470$) without improving minority detection over linear baselines.
* **Dataset Constraints:** Static tabular variables account for limited variance in resignation decisions. Model performance capped at ~0.58 F1 without external market or real-time data.


