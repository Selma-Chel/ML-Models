# House Price Prediction Using Machine Learning

## Overview

This project predicts residential house prices using the Ames Housing dataset from Kaggle. The goal is to build an end-to-end machine learning pipeline capable of handling missing values, categorical variables, skewed numerical features, and feature engineering while producing accurate house price predictions.

The project explores multiple regression models, feature engineering techniques, preprocessing strategies, and hyperparameter tuning to identify the best-performing solution.

---

## Dataset

**Source:** Kaggle House Prices: Advanced Regression Techniques

The dataset contains information about residential properties, including:

* Construction year
* Living area
* Basement characteristics
* Garage information
* Exterior materials
* Quality ratings
* Neighborhood information

The target variable is:

* **SalePrice** – the final sale price of each house

---

## Data Cleaning

The following preprocessing steps were applied:

* Removed duplicate rows
* Removed the `Id` column
* Dropped features with a large proportion of missing values:

  * PoolQC
  * Fence
  * MiscFeature
  * Alley

Missing values were handled using Scikit-Learn pipelines:

* Median imputation for numerical variables
* Most frequent value imputation for categorical variables
* Constant-value imputation for specific features such as `GarageYrBlt`

---

## Feature Engineering

Several engineered features were created to reduce redundancy and combine related information.

### Basement Finish Quality

Combined:

* BsmtFinType1
* BsmtFinType2

into:

* `QualBsmtFinType`

### Garage Quality

Combined:

* GarageCond
* GarageQual

into:

* `OverallGarageQ`

### Bathroom Features

Combined:

* FullBath
* BsmtFullBath

into:

* `AllFullbath`

Combined:

* HalfBath
* BsmtHalfBath

into:

* `AllHalfbath`

### Exterior Features

Created:

* `exterior_material`
* `Exteriortype`

using exterior wall materials.

### Basement Area

Combined:

* BsmtFinSF1
* BsmtFinSF2

into:

* `TotalBsmtFinsf`

### Additional Experiments

Several additional engineered features were tested, including:

* House age
* Remodeling age
* Total square footage

These features did not improve validation performance and were excluded from the final model.

---

## Feature Preprocessing

### Numerical Features

Numerical variables were divided into:

* Skewed numerical features
* Non-skewed numerical features

### Skewed Features

Applied:

* Median imputation
* Yeo-Johnson transformation

### Non-Skewed Features

Applied:

* Median imputation
* Robust scaling

### Categorical Features

Separated into:

#### Nominal Categories

Processed using:

* Most frequent imputation
* One-Hot Encoding

#### Ordinal Categories

Processed using:

* Most frequent imputation
* Ordinal Encoding

with manually defined category orderings.

---

## Target Transformation

The target variable (`SalePrice`) was transformed using:

```python
np.log1p(SalePrice)
```

This reduced the effect of extreme house prices and improved model performance.

Predictions were transformed back to the original scale using:

```python
np.expm1()
```

---

## Models Evaluated

The following models were tested:

* Linear Regression
* Ridge Regression
* ElasticNet
* Support Vector Regression (SVR)
* XGBoost Regressor

Hyperparameter tuning was performed using GridSearchCV.

---

## Final Model

**ElasticNet Regression**

The final model was selected after comparing multiple algorithms and tuning hyperparameters using GridSearchCV.

The complete preprocessing workflow and model were combined into a single Scikit-Learn Pipeline.

---

## Results

### Kaggle Public Leaderboard Score

**0.12555**

### Key Findings

* Target transformation significantly improved predictive performance.
* Separating ordinal and nominal categorical variables improved preprocessing quality.
* Additional feature engineering did not always improve performance.
* ElasticNet outperformed more complex models tested in this project.
* Proper preprocessing contributed more to performance gains than increasing model complexity.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost

---

## Lessons Learned

This project reinforced several important machine learning concepts:

* More features do not necessarily lead to better performance.
* Feature engineering must be validated experimentally.
* Proper preprocessing can have a larger impact than model selection.
* Target transformations can improve regression performance on skewed data.
* Cross-validation and hyperparameter tuning help improve model generalization.
