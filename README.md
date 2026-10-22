# Machine Learning Projects

A collection of nine end-to-end machine learning notebooks covering classification, regression, computer vision, NLP, time-series and automated pipelines.

## Overview

Each notebook is self-contained: it loads its own data from `Data/`, preprocesses it, trains one or more models and reports evaluation metrics. The projects progress from a from-scratch implementation of linear regression to a fine-tuned BERT model and a reusable automated ML pipeline.

## Repository Structure

```
.
├── Data/                      # Datasets used by the notebooks
├── Notebooks/
│   ├── artifacts/             # Output of notebook 9 (model.joblib, metadata.json)
│   ├── 1-Iris_classification.ipynb
│   ├── 2-Linear_Regression_Scratch.ipynb
│   ├── 3-Titanic.ipynb
│   ├── 4-Housing_price_predictor.ipynb
│   ├── 5-Cat_Dog.ipynb
│   ├── 6-Sentiment_analysis.ipynb
│   ├── 7-Customer_Churn_Prediction.ipynb
│   ├── 8-Stock_Price_Predicition.ipynb
│   └── 9-ML_Pipeline.ipynb
├── LICENSE
└── README.md
```

## Projects

### 4.1 Iris Classification

#### Overview
Classifies iris flowers into three species from four measurements using a Decision Tree.

#### Dataset
`Data/Iris.csv`. The `Id` column is dropped. No cleaning is required.

#### Exploratory Data Analysis
Summary statistics, feature histograms and a scatter matrix on the training set.

#### Train/Test Split
80/20 stratified split on `Species` (`random_state=42`).

#### Model
`DecisionTreeClassifier` with default parameters.

#### Evaluation
| Metric | Value |
|---|---|
| Train accuracy | 100% |
| Test accuracy | 96.7% (29 of 30 correct) |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Iris-setosa | 1.00 | 1.00 | 1.00 |
| Iris-versicolor | 1.00 | 0.90 | 0.95 |
| Iris-virginica | 0.91 | 1.00 | 0.95 |

Confusion matrix (rows = actual, columns = predicted):

```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

The only error is one versicolor predicted as virginica. The test set has just 30 samples, so a single mistake moves accuracy by 3.3 points.

#### Feature Importance and Tree Visualization
Bar chart of feature importances and a plot of the fitted tree.

#### How to Run
Open `Notebooks/1-Iris_classification.ipynb` and run all cells.

---

### 4.2 Linear Regression from Scratch

#### Overview
Linear regression implemented with NumPy and gradient descent, with no machine learning library.

#### Dataset
`Data/Salary Data.csv`: `YearsExperience` (feature) and `Salary` (target).

#### Mathematical Intuition
- Hypothesis: `f(x) = w·x + b`
- Cost: `J(w, b) = (1 / 2m) · Σ (f(xᵢ) − yᵢ)²`
- Gradients: `∂J/∂w = (1/m) · Σ (f(xᵢ) − yᵢ)·xᵢ` and `∂J/∂b = (1/m) · Σ (f(xᵢ) − yᵢ)`
- Update rule: `w ← w − α·∂J/∂w`, `b ← b − α·∂J/∂b`

#### Implementation
Three functions: `cost_function`, `gradient_function` and `gradient_descent`. Both parameters start at 0.

#### Hyperparameters
| Parameter | Value |
|---|---|
| Learning rate (α) | `[FILL]` |
| Iterations | 20,000 |

#### Results
Final parameters: `w = 5986.25`, `b = 45666.16`. The notebook plots the fitted line over the data.

<!-- Verify these against np.polyfit(x_train, y_train, 1) before publishing. -->

#### How to Run
Open `Notebooks/2-Linear_Regression_Scratch.ipynb` and run all cells.

---

### 4.3 Titanic Survival Prediction

#### Overview
Predicts passenger survival on the Titanic (Kaggle competition format).

#### Dataset
`Data/titanic/train.csv` and `Data/titanic/test.csv`.

#### Data Cleaning
Missing `Age` values are filled with the median. `Name`, `Ticket` and `Cabin` are dropped.

#### Feature Selection
`Pclass`, `Sex`, `SibSp`, `Parch`, with `Sex` one-hot encoded via `pd.get_dummies`.

#### Model
`RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)`.

#### Results
Kaggle public score: **0.77511**. Predictions are written to `submission.csv`.

#### How to Run
Open `Notebooks/3-Titanic.ipynb` and run all cells.

---

### 4.4 Housing Price Predictor

#### Overview
Predicts house sale prices from property attributes (Kaggle House Prices format) with a weighted ensemble of three tuned models.

#### Dataset
`Data/house/train.csv` and `Data/house/test.csv`.

#### Feature Engineering
- Binary `has_*` flags for `Alley`, `BsmtQual`, `PoolQC`, `Fence`, `MiscFeature` and `GarageYrBlt`.
- `TotalSF` = basement + 1st floor + 2nd floor area.
- `HouseAge` = `YrSold` − `YearBuilt`.

#### Handling Missing Values
Missing `GarageYrBlt` is filled with `YearBuilt`. Other numeric nulls become 0 and categorical nulls become `"None"`, since most nulls mean the feature is absent.

#### Encoding
Quality and condition columns are ordinal-encoded with explicit category order. The remaining categorical columns are one-hot encoded (`handle_unknown='ignore'`).

#### Feature Scaling
`StandardScaler`, fitted on the training set only.

#### Models and Hyperparameter Tuning
`GridSearchCV` (5-fold, scored on negative RMSE) for each model:

| Model | Tuned parameters |
|---|---|
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| Ridge | `alpha` |

#### Weighted Ensemble
Final prediction = `0.5 × XGBoost + 0.3 × Random Forest + 0.2 × Ridge`.

#### Results
Kaggle leaderboard score (RMSE on log prices, lower is better): **0.12345** for the best submission, and 0.12443 for the notebook's version 6 run.

#### How to Run
Open `Notebooks/4-Housing_price_predictor.ipynb`. The grid searches are slow, so expect a long run on CPU.

---

### 4.5 Cat vs Dog Classifier

#### Overview
Binary image classifier that separates cats from dogs with a convolutional neural network.

#### Dataset
`Data/catdog/training_set` and `Data/catdog/test_set`, one subfolder per class.

#### Data Loading and Augmentation
Images are resized to 180×180 and batched in 32s. The training folder is split 80/20 into training and validation (`seed=123`). Augmentation: random horizontal flip, rotation (0.1) and zoom (0.1). Pixels are rescaled to [0, 1].

#### Model Architecture
Three `Conv2D` + `MaxPooling2D` blocks (32, 64, 128 filters), `Flatten`, `Dense(128)`, `Dropout(0.4)` and a sigmoid output.

#### Training
Adam optimizer, binary cross-entropy, up to 40 epochs. `EarlyStopping` (patience 5, monitors `val_loss`, restores best weights). `ModelCheckpoint` saves `best_model.keras` by `val_accuracy`.

#### Evaluation
Test accuracy and loss, training curves, confusion matrix and classification report.

#### Results
Test accuracy: `[FILL]`. Test loss: `[FILL]`.

#### How to Run
Open `Notebooks/5-Cat_Dog.ipynb`. A GPU speeds up training noticeably.

---

### 4.6 Sentiment Analysis

#### Overview
Classifies text sentiment by fine-tuning `bert-base-uncased`.

#### Dataset
`Data/sentiment_analysis.csv` with `text` and `sentiment` columns.

#### Data Splitting
Stratified 70% train, 15% validation, 15% test (`random_state=42`).

#### Text Preprocessing
Lowercasing and URL removal only. Stop-word removal and lemmatization are not applied, because BERT uses the raw token context.

#### Tokenization
`bert-base-uncased` tokenizer with truncation at 256 tokens.

#### Model (BERT Fine-Tuning)
`AutoModelForSequenceClassification` with the number of labels taken from the data.

#### Training Configuration
| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |

Training uses the Hugging Face `Trainer`, evaluated on the validation set each epoch.

#### Evaluation
Accuracy plus weighted precision, recall and F1, a confusion matrix and a per-class classification report on the test set.

#### Results
| Metric | Value |
|---|---|
| Test accuracy | 0.69 |
| Weighted precision | 0.71 |
| Weighted recall | 0.69 |
| Weighted F1 | 0.69 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.76 | 0.65 | 0.70 | 20 |
| neutral | 0.63 | 0.80 | 0.71 | 30 |
| positive | 0.75 | 0.60 | 0.67 | 25 |

Confusion matrix (rows = actual, columns = predicted; order: negative, neutral, positive):

```
[[13  5  2]
 [ 3 24  3]
 [ 1  9 15]]
```

The test set has only 75 samples, so treat these numbers as indicative, not precise. Most errors involve the neutral class.

#### How to Run
Install the spaCy model first (`python -m spacy download en_core_web_sm`). A GPU is strongly recommended.

---

### 4.7 EV Purchase Prediction

#### Overview
Entry for the Kaggle competition `[FILL: competition name]`. Predicts whether a person will buy an electric vehicle (`Will_Buy_EV`) with a neural network and a tuned decision threshold.

#### Dataset
`Data/ev-purchase/train.csv` and `Data/ev-purchase/test.csv`.

#### Preprocessing
- Null and duplicate checks.
- Ordinal encoding for `Home_Charging_Possible`, `Subsidy_Available` and `Range_Anxiety_Level`.
- One-hot encoding for the remaining categorical columns.
- Label encoding of the target and standard scaling of features.

#### Outlier Handling
IQR check on all numeric columns. `Annual_Income_USD` and `Daily_Commute_km` are clipped to the 1.5×IQR bounds, using bounds computed on the training set.

#### Model Architecture
`Dense(64)` → `Dropout(0.3)` → `Dense(32)` → `Dropout(0.3)` → sigmoid output. Adam optimizer, binary cross-entropy, AUC and accuracy metrics. `EarlyStopping` on validation AUC (patience 5, max 100 epochs).

#### Threshold Tuning
The decision threshold is chosen on the validation set to maximize F1, then applied to the test predictions.

#### Results
Best leaderboard submission (metric: `[FILL]`): **0.93766** public, **0.93722** private. Best validation threshold: `[FILL]`, with F1 of `[FILL]` at that threshold.

#### How to Run
Open `Notebooks/7-Customer_Churn_Prediction.ipynb`. Predictions are written to `submission.csv`.

---

### 4.8 Stock Movement Prediction

#### Overview
Predicts whether a stock's adjusted close will be higher the next trading day (a direction label, not a price) for Amazon, Apple, Google, Microsoft and Netflix.

#### Data Source
Daily price CSVs in `Data/stock_data/`, one file per company. Source: `[FILL]`.

#### Feature Engineering
Daily return, 5/10/20-day moving-average ratios, 14-day RSI, 10 and 20-day volatility, volume change and a one-hot company indicator.

#### Chronological Split
| Set | Period |
|---|---|
| Train | up to 2018-12-31 |
| Validation | 2019-01-01 to 2021-06-30 |
| Test | from 2021-07-01 |

The split is by date, not random, so no future data leaks into training.

#### Sequence Construction
Rolling 30-day windows are built per company, so a window never mixes two stocks. The scaler is fitted on the training set only.

#### Models
- Logistic Regression (baseline, flattened windows)
- XGBoost (200 trees, depth 4, learning rate 0.05)
- LSTM: `LSTM(64)` → `Dense(32)` → sigmoid, Adam (lr 0.0005), early stopping on validation loss

#### Evaluation
Accuracy and classification report. Logistic Regression and XGBoost are scored on the validation set. The LSTM is scored on the test set.

#### Results
| Model | Validation accuracy | Test accuracy |
|---|---|---|
| Logistic Regression | `[FILL]` | `[FILL]` |
| XGBoost | `[FILL]` | `[FILL]` |
| LSTM | not reported | 48.9% |
| Always predict the majority class (label 0) | not reported | 51.0% |

LSTM test report: precision 0.50 / recall 0.35 for label 0 and precision 0.48 / recall 0.63 for label 1, over 2,010 test sequences. The LSTM scores below the majority-class baseline, so it shows no edge over chance on this data.

#### Limitations
- Predicts direction only, not price or magnitude.
- Trained on five large-cap tech stocks, so results may not generalize to other sectors.
- Ignores transaction costs and slippage, so accuracy does not equal trading profit.
- Uses daily price and volume only, with no news or fundamentals.

#### How to Run
Open `Notebooks/8-Stock_Price_Predicition.ipynb` and run all cells.

---

### 4.9 ML Pipeline

#### Overview
An interactive pipeline that takes any CSV, asks for the target and task type, then cleans, benchmarks, tunes and saves the best model.

#### Pipeline Stages
1. Load the file (delimiter auto-detected).
2. Inspect: shape, column types, nulls, duplicates, constant columns.
3. Analyze the target: distribution, skew, class balance.
4. Confirm the task (regression or classification) with soft warnings on suspicious choices.
5. Clean: drop rows with a missing target, duplicates, constant, mostly-null, ID-like and duplicated columns.
6. Split 80/20 (stratified for classification).
7. Preprocess: imputation, scaling, one-hot encoding.
8. Benchmark all models with 5-fold cross-validation against a dummy baseline.
9. Tune the top 3 models with `RandomizedSearchCV` (30 iterations).
10. Evaluate the best model on the held-out test set and save it.

#### Supported Models
| Task | Models |
|---|---|
| Regression | Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost |
| Classification | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |

Primary metric: RMSE for regression, weighted F1 for classification.

#### Interactive Inputs
Set `FILE_PATH` in the config cell. At runtime the notebook prompts for the target column and the task type (`r` or `c`).

#### Output Artifacts
Saved to `Notebooks/artifacts/`:
- `model.joblib`: the best fitted pipeline.
- `metadata.json`: task, target, model name, test scores, best parameters, feature list, random state and scikit-learn version.

#### Results
Run on the regression house-price dataset (`Pipeline_data/Regression/house/train.csv`), with the sale price as the target. Best model after tuning: **XGBoost**.

Test set results:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (predict the mean) | 87,619.03 | 62,575.93 | -0.0009 |
| XGBoost (tuned) | 26,083.47 | 16,779.96 | 0.9113 |

Top 3 models after hyperparameter tuning (5-fold CV RMSE on the training set):

| Model | Untuned RMSE | Tuned RMSE |
|---|---|---|
| XGBoost | 30,582.33 | 25,844.53 |
| Gradient Boosting | 27,669.03 | 26,691.62 |
| Random Forest | 30,611.01 | 29,525.94 |

#### How to Run
Run the notebook and answer the prompts. Load the saved model later with `joblib.load("artifacts/model.joblib")`, using the same scikit-learn version recorded in `metadata.json`.

## Tech Stack

- **Language:** Python `[FILL: version]`
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, seaborn
- **Classical ML:** scikit-learn, XGBoost
- **Deep learning:** TensorFlow / Keras, PyTorch
- **NLP:** Hugging Face Transformers, spaCy
- **Utilities:** joblib, SciPy, Jupyter

## Datasets

| Project | Files | Source |
|---|---|---|
| Iris | `Iris.csv` | `[FILL]` |
| Linear Regression | `Salary Data.csv` | `[FILL]` |
| Titanic | `titanic/` | `[FILL]` |
| Housing | `house/` | `[FILL]` |
| Cat vs Dog | `catdog/` | `[FILL]` |
| Sentiment | `sentiment_analysis.csv` | `[FILL]` |
| EV Purchase | `ev-purchase/` | Kaggle competition data: `[FILL]` |
| Stocks | `stock_data/` | `[FILL]` |
| ML Pipeline | `Pipeline_data/` | `[FILL]` |

Datasets belong to their original owners and follow their own licenses, which are separate from this repository's license.

## Installation

```bash
git clone [FILL: repo URL]
cd [FILL: repo name]

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost \
            tensorflow torch transformers spacy joblib jupyter
python -m spacy download en_core_web_sm
```

## How to Run the Notebooks

1. Start Jupyter **from inside the `Notebooks/` folder**, because every notebook reads data through the relative path `../Data/`:
   ```bash
   cd Notebooks
   jupyter lab
   ```
2. Open a notebook and run the cells top to bottom.
3. Notebook 9 asks for input in the terminal or cell prompt, so answer when it appears.
4. Notebooks 5 and 6 train neural networks and run much faster on a GPU.

## Results Summary

| # | Project | Task | Model | Metric | Score |
|---|---|---|---|---|---|
| 1 | Iris Classification | Classification | Decision Tree | Test accuracy | 96.7% |
| 2 | Linear Regression from Scratch | Regression | Gradient descent | Final w, b | 5986.25, 45666.16 |
| 3 | Titanic | Classification | Random Forest | Kaggle public score | 0.77511 |
| 4 | Housing Price Predictor | Regression | XGB + RF + Ridge ensemble | Kaggle score (log RMSE) | 0.12345 |
| 5 | Cat vs Dog | Image classification | CNN | Test accuracy | `[FILL]` |
| 6 | Sentiment Analysis | NLP classification | BERT | Test accuracy / weighted F1 | 0.69 / 0.69 |
| 7 | EV Purchase Prediction | Classification | Neural network | Kaggle public / private | 0.93766 / 0.93722 |
| 8 | Stock Movement Prediction | Time-series classification | LSTM | Test accuracy | 48.9% |
| 9 | ML Pipeline | Regression | XGBoost (auto-selected) | Test RMSE / R² | 26,083 / 0.911 |

## License

This project is licensed under the `[FILL: MIT]` License. See the [LICENSE](LICENSE) file for details. The license covers the code in this repository, not the third-party datasets.