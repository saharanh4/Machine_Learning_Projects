## 1.) Iris Classification
### Overview
This project is made to learn about machine learning using project based learning it is a very basic project

### Objective
I want to classify flowers on the following parameters so that when a new data is introduced machine can easily classify on the basis of.
- Petal Width 
- Petal Length
- Sepal Width
- Sepal Length

### Dataset
Iris Classification data is use 

### Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
    - train_test_split
    - tree
    - Metrics
        - Classification report 
        - Confusion matrix
    - tree
        - Plot Tree
        - Decision Tree Classifier

### Model/Algorithm & Result
Decision Tree Classifier is used to predict the outcome which yield a good accuracy of 96.6 Percent 

### Future Improvements
- Accuracy can be further improved by changing used model
- Pipelines can be created too

### Learning Outcomes
- Basic use of models 
- Use of different libraries in a machine learning workflow environment 

## 2.) Salary Prediction — Linear Regression from Scratch
A simple linear regression model implemented from scratch (no `sklearn.linear_model`) to predict salary from years of experience, using manually coded cost function and batch gradient descent.

### Overview

This project predicts `Salary` from `YearsExperience` using univariate linear regression. The gradient descent update rule, cost function, and training loop are all implemented manually in NumPy/pandas to demonstrate the underlying mechanics rather than relying on a library like scikit-learn.

### Features

- Manual implementation of the squared-error cost function
- Manual implementation of the gradient descent update rule
- Data visualization: raw scatter plot and fitted regression line
- No black-box ML library used for training — pure math implementation

### Tech Stack

- Python 3.x
- pandas / NumPy
- Matplotlib

### Dataset

`Salary Data.csv` — two columns:
- `YearsExperience` (feature)
- `Salary` (target)

### Model

Simple linear model: `Salary = w * YearsExperience + b`

Trained via batch gradient descent:
- **Learning rate:** 0.0000199
- **Iterations:** 20,000
- **Result:** `w ≈ 5986.25`, `b ≈ 45666.16`

### Future Improvements

- Fix the gradient return/unpack order and re-validate `w`/`b`
- Fix the cost function precedence bug
- Add a cost-vs-iteration plot to visually confirm convergence
- Validate against `sklearn.linear_model.LinearRegression` as ground truth
- Add train/test split and report R² / MSE on held-out data


## 3.) Titanic Survival Prediction — Random Forest

A binary classification model predicting Titanic passenger survival using a Random Forest classifier, based on the classic Kaggle Titanic dataset.

### Overview

This project trains a Random Forest on the Kaggle Titanic training set and generates survival predictions for the test set, using a small set of demographic/family features.

### Features

- Basic feature engineering (missing value imputation)
- One-hot encoding of categorical features
- Random Forest classifier with fixed depth to limit overfitting
- Predictions generated for Kaggle's held-out test set

### Tech Stack

- Python 3.x
- pandas
- scikit-learn (`RandomForestClassifier`)

### Dataset

Kaggle's Titanic dataset (`train.csv`, `test.csv`), containing passenger details: class, sex, age, siblings/spouses aboard, parents/children aboard, fare, cabin, ticket, and (for training data) survival outcome.

### Features Used for Training

```
Pclass, Sex, SibSp, Parch
```

One-hot encoded via `pd.get_dummies`.

### Model

- **Algorithm:** Random Forest Classifier
- **Trees:** 100
- **Max depth:** 5
- **Random state:** 42 (for reproducibility)


**requirements.txt**
```
pandas
scikit-learn
```

### Future Improvements

- Add a real validation split and report actual, reproducible accuracy
- Use `Age` (now that it's imputed) and `Fare` as additional features
- Try feature engineering on `Cabin` (deck letter) and `Name` (title extraction) instead of dropping them outright
- Tune `n_estimators` / `max_depth` via cross-validation or grid search
- Generate and save a Kaggle-format `submission.csv`


## 4.) House Prices Prediction 

A regression pipeline predicting house sale prices (Kaggle's "House Prices: Advanced Regression Techniques") using a weighted ensemble of XGBoost, Random Forest, and Ridge regression, with hyperparameters tuned via `GridSearchCV`.

### Overview

This project builds a full tabular regression pipeline: missing-value handling, manual feature engineering, ordinal + one-hot encoding of categorical features, feature scaling, hyperparameter search across three models, and a manually-weighted ensemble of their predictions.

**Kaggle result:** RMSE (log SalePrice) ≈ 0.12345, Rank ≈ 939

### Features

- Missing-value visualization (bar chart of top 20 columns by null count)
- Custom feature engineering: `has_<feature>` flags, `TotalSF`, `HouseAge`
- Domain-aware ordinal encoding for quality-scale features (`Po`→`Ex`), fence type, garage finish, basement exposure, and finish type
- One-hot encoding for remaining nominal categorical features
- Feature scaling on continuous/ordinal columns
- Hyperparameter tuning via `GridSearchCV` (5-fold CV, RMSE scoring) for XGBoost, Random Forest, and Ridge
- Weighted ensemble of the three tuned models' predictions

### Tech Stack

- Python 3.x
- pandas / Matplotlib
- scikit-learn (`OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`, `Ridge`, `RandomForestRegressor`, `GridSearchCV`)
- XGBoost (`XGBRegressor`)

### Dataset

Kaggle's "House Prices: Advanced Regression Techniques" (`train.csv`, `test.csv`) — 79 explanatory variables describing residential homes in Ames, Iowa. Target: `SalePrice`.

### Pipeline

1. **Null-value inspection** — bar chart of the top 20 columns by missing-value count
2. **Feature engineering** — presence flags for sparse features (`Alley`, `BsmtQual`, `PoolQC`, `Fence`, `MiscFeature`, `GarageYrBlt`), plus `TotalSF` (total square footage) and `HouseAge` (years since built)
3. **Null handling** — `GarageYrBlt` imputed from `YearBuilt`; categorical nulls filled with `'None'`; numeric nulls filled with `0`
4. **Ordinal encoding** — quality/condition-scale columns mapped to an explicit ordered scale (`None < Po < Fa < TA < Gd < Ex`, and dataset-specific scales for fence, garage finish, basement exposure/finish type)
5. **One-hot encoding** — remaining nominal categorical columns
6. **Scaling** — `StandardScaler` applied to continuous/ordinal columns
7. **Model tuning** — `GridSearchCV` (5-fold, RMSE) for XGBoost, Random Forest, and Ridge independently
8. **Ensemble** — final prediction = `0.5 * XGBoost + 0.3 * RandomForest + 0.2 * Ridge`

### Usage

Place `train.csv` and `test.csv` in the `Data/house/` directory, then run:

```bash
python house_price_ensemble.py
```

This generates `submission.csv` in Kaggle submission format (`Id`, `SalePrice`).

### Future Improvements

- Confirm the `'str'` dtype checks are actually matching categorical columns (see Known Issues)
- Log-transform `SalePrice` and skewed numeric features before training (standard practice for this dataset)
- Tune ensemble weights against a proper out-of-fold validation scheme instead of hardcoding them
- Try target encoding for high-cardinality categorical features (e.g. `Neighborhood`)
- Add SHAP or feature importance analysis to explain model predictions
