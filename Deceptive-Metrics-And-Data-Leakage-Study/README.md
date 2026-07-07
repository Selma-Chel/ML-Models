#  Case Study: Deceptive Metrics, Target Leakage, and Class Imbalance in Seattle Building Permits

[![Dataset Source: Data.gov](https://catalog.data.gov/dataset/building-permits?from_hint=eyJxIjoiYnVpbGRpbmcgcGVybWl0cyIsInNvcnQiOiJyZWxldmFuY2UifQ%3D%3D)
Author: Selma Chelbabi

##  Executive Summary
Most machine learning portfolios only showcase flawless models hitting 95%+ accuracy on highly sanitized datasets. **This project is an intentional post-mortem analysis of why real-world machine learning pipelines fail.** Using a public dataset of city building permits from Data.gov, this case study exposes how a flawed predictive objective, severe class imbalance, and hidden target leakage can combine to give a deceptive illusion of performance. It maps the diagnostic journey from a raw python pipeline to uncovering why a model hitting **77% accuracy** is actually completely non-viable for production deployment.

---

## 📊 The Core Pipeline Structure

The project notebook is structured into two distinct operational phases:

1. **Phase 1: Pipeline Execution (The Engineering):** * Automated missingness filters (dropping columns with >40% missing values).
   * Robust date extraction and cycle preprocessing via Scikit-Lego's `RepeatingBasisFunction`.
   * Class-separated preprocessing pipelines via Scikit-Learn `ColumnTransformer`.
   * Ensembled training using an optimized `XGBRFClassifier` evaluated via Randomized Grid Search.
2. **Phase 2: Diagnostic Post-Mortem (The Analysis):**
   * A granular teardown explaining how mathematical "gravity" and structural column masking manipulated the model's metrics.

---

##  Top 3 Engineering Takeaways Explored

### 1. The Accuracy Trap
While an overall validation accuracy of **77%** sounds decent, a baseline `DummyClassifier` hits **69.6%** simply by guessing the majority class every single time. The model's **Macro Average F1-Score of 0.30** exposes that the algorithm completely fails to generalize across the 23 granular permit statuses.

### 2. Target Leakage vs. Imputation Masking
Features like `IssuedDateYear` and `ExpiresDateYear` represent severe **Target Leakage**—they provide data that would physically only exist *after* a permit reaches its terminal state. However, because missing values for early-stage permits were imputed with historical medians, this leakage was accidentally "masked," confusing the model splits and capping its ability to perfectly overfit.

### 3. Structural Resilience via Feature Subsampling
The model resisted exploding into 100% training overfitting due to the architectural mechanics of the `XGBRFClassifier`. By enforcing strict **feature subsampling at every node split**, individual decision forks were regularly blinded to the leaky date columns, forcing the ensemble to train on sub-optimal, non-leaky features.

---

##  How to Reframe the Business Objective
This post-mortem concludes that **no amount of hyperparameter tuning can fix a fundamentally flawed predictive objective.** To turn this pipeline into an impactful business tool, the engineering strategy must be pivoted:

* **Eradicate the Leakage:** Completely drop all retrospective temporal milestones (`IssuedDate`, `ExpiresDate`) and rely solely on inputs available at the split-second of filing.
* **Target Class Consolidation:** Collapse the 23 volatile status classes down into 3 structurally sound macro-categories: `Approved/Completed`, `In-Progress/Delayed`, and `Canceled/Withdrawn`.

---

## Tech Stack & Packages Used
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Feature Engineering:** `feature_engine` (CountFrequencyEncoder), `scikit-lego` (RepeatingBasisFunction)
* **Visualization:** `matplotlib`
