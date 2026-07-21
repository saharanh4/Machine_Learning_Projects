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

### ⚠️ Known Issues (Please Read Before Using)

This is a learning project and currently has two unresolved bugs:

1. **Swapped gradient return values.** `gradient_function` returns `(dc_db, dc_dw)`, but `gradient_descent` unpacks the result as `dc_dw, dc_db = gradient_function(...)`. This means `w` is actually updated using `dc_db` and `b` is updated using `dc_dw`. The model still trains and produces *a* line, but the reported `w`/`b` should not be trusted as the true least-squares fit until this is fixed and verified against a library implementation (e.g. `sklearn.linear_model.LinearRegression`).
2. **Operator precedence bug in `cost_function`.** `(1/2*m) * cost_sum` evaluates as `(1/2)*m`, not `1/(2*m)`. This function isn't currently called during training, but will produce an incorrect cost value if the commented-out print statement is re-enabled.

**Recommended fix before relying on results:**
```python
dc_dw, dc_db = gradient_function(x, y, w, b)  # match the actual return order

def cost_function(x, y, w, b):
    ...
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost
```

**requirements.txt**
```
pandas
numpy
matplotlib
```

### Future Improvements

- Fix the gradient return/unpack order and re-validate `w`/`b`
- Fix the cost function precedence bug
- Add a cost-vs-iteration plot to visually confirm convergence
- Validate against `sklearn.linear_model.LinearRegression` as ground truth
- Add train/test split and report R² / MSE on held-out data
