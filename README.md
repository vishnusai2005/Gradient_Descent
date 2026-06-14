# 🚗 Car Price Prediction — Gradient Descent, Regularization & Regression Diagnostics

> **End-to-end regression pipeline** on the CarDekho dataset covering custom Gradient Descent implementation, Linear Regression diagnostics (Normality of Residuals, Multicollinearity via VIF, Homoscedasticity), Ridge Regression, and Lasso Regression with feature selection.

---

##  Problem Statement

Predict the **selling price of a used car** based on its age, kilometers driven, fuel type, transmission type, seller type, and ownership history — using the [CarDekho Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

---

##  Table of Contents

- [Dataset Overview](#-dataset-overview)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering & Encoding](#-feature-engineering--encoding)
- [Model Training — Linear Regression](#-model-training--linear-regression)
- [Custom Gradient Descent](#-custom-gradient-descent-from-scratch)
- [Assumptions of Linear Regression](#-assumptions-of-linear-regression)
  - [Normality of Residuals](#1-normality-of-residuals)
  - [No Multicollinearity (VIF)](#2-no-multicollinearity)
  - [Homoscedasticity](#3-homoscedasticity)
- [Ridge Regression (L2 Regularization)](#-ridge-regression-l2-regularization)
- [Lasso Regression (L1 Regularization)](#-lasso-regression-l1-regularization)
- [Model Comparison](#-model-comparison)
- [Key Takeaways](#-key-takeaways)

---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| **Source** | CarDekho (CSV) |
| **Target Variable** | `selling_price` |
| **Original Features** | `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner` |
| **Engineered Feature** | `Age` = `max(year) + 1 − year` |

The `name` column was dropped due to **1,491 unique categories** — too high cardinality relative to the dataset size.

---

## 🛠️ Tech Stack

```
Python 3.x | NumPy | Pandas | Matplotlib | Seaborn | Scikit-learn | SciPy | Statsmodels
```

---

##  Project Workflow

```
Raw Data
   │
   ▼
Data Preparation  ──► Drop 'name', Engineer 'Age', Handle Nulls & Duplicates
   │
   ▼
EDA  ──► Countplots, Boxplots, Histograms, KDE Plots, Scatter Plots, Correlation Heatmap
   │
   ▼
Feature Engineering  ──► One-Hot Encoding (drop_first=True) → Correlation Heatmap
   │
   ▼
Train-Test Split (80/20)  ──► StandardScaler
   │
   ▼
Model Training  ──► Linear Regression (sklearn) + Custom Gradient Descent (NumPy)
   │
   ▼
Regression Diagnostics  ──► Linearity · Normality of Residuals · VIF · Homoscedasticity
   │
   ▼
Regularization  ──► Ridge (α=10) · Lasso (α=10,000)
   │
   ▼
Evaluation  ──► MAE · MSE · RMSE · R² Score
```

---

## 📈 Exploratory Data Analysis

### Categorical Features
- **Fuel**: 5 categories — Diesel has the highest frequency; CNG, LPG, and Electric have the least.
- **Seller Type**: 3 categories — Individual sellers dominate; Trustmark Dealers are fewest.
- **Transmission**: 2 categories — Manual overwhelmingly outnumbers Automatic.
- **Owner**: 5 categories — First Owner accounts for the majority of listings.

### Numerical Features
Histogram, Boxplot, and KDE plots were generated for `Age`, `selling_price`, and `km_driven`:
- `selling_price` is **right-skewed**, with a few high-value outliers.
- `km_driven` is also right-skewed, indicating most cars have moderate mileage.
- `Age` distribution is spread across older vehicles in inventory.

### Bivariate Analysis
- **Age vs. Selling Price**: Newer cars (low Age) command significantly higher prices, especially First Owners.
- **km_driven vs. Selling Price**: Lower-mileage cars tend to sell for more, though with high variance.

---

##  Feature Engineering & Encoding

```python
# Year → Age (reverses the ordering intuitively)
data.insert(0, "Age", data['year'].max() + 1 - data['year'])
data.drop('year', axis=1, inplace=True)

# One-Hot Encoding with drop_first to avoid dummy variable trap
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
df_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True).astype(int)
```

A **correlation heatmap** was plotted post-encoding to understand linear relationships between all features and the target `selling_price`.

---

##  Model Training — Linear Regression

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

linear_reg = LinearRegression()
linear_reg.fit(x_train_scaled, y_train)
```

**Evaluation Metrics Used:**

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error |
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error |
| **R² Score** | Proportion of variance explained by the model |

---

## ⚙️ Custom Gradient Descent (From Scratch)

A vectorized Gradient Descent optimizer was implemented from scratch using NumPy — without any ML library — to demonstrate the core mathematical mechanics of weight optimization.

```python
def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    m = x.shape[0]          # Number of samples
    n = x.shape[1]          # Number of features
    weights = np.zeros(n)   # Initialize weights to zero
    cost_history = []

    for epoch in range(epochs):
        predictions = np.dot(x, weights)           # ŷ = Xw
        errors      = predictions - y              # Residuals
        gradients   = (1/m) * np.dot(x.T, errors) # ∂J/∂w
        weights    -= learning_rate * gradients    # Weight update
        cost        = (1/(2*m)) * np.sum(errors**2)  # MSE Cost
        cost_history.append(cost)

    return weights, cost_history
```

**Key Concepts Implemented:**

| Concept | Formula |
|---|---|
| **Prediction** | `ŷ = Xw` |
| **Gradient** | `(1/m) · Xᵀ(ŷ − y)` |
| **Weight Update** | `w = w − α · ∇J(w)` |
| **Cost (MSE)** | `(1/2m) · Σ(ŷ − y)²` |

A bias term (column of ones) was prepended to `x_train_scaled` before feeding it into the optimizer:

```python
x_train_np = np.hstack((np.ones((x_train_scaled.shape[0], 1)), x_train_scaled))
weights, cost_history = gradient_descent(x_train_np, y_train_np, learning_rate=0.01, epochs=1000)
```

**Cost curve** was plotted to confirm the model converges smoothly over 1,000 epochs — a classic downward-sloping loss curve indicating stable gradient updates with `lr=0.01`.

---

##  Assumptions of Linear Regression

Post-training diagnostic checks were performed to validate that the fitted model satisfies the core assumptions of Ordinary Least Squares (OLS) regression.

---

### 1. Normality of Residuals

**Why it matters:** OLS inference (p-values, confidence intervals) is valid only when residuals are approximately normally distributed.

**How it was checked:**

```python
import scipy.stats as stats

def check_normality_of_residuals(model, x_test, y_test):
    residuals_data = (y_test - model.predict(x_test))

    # Histogram with KDE overlay
    sns.histplot(residuals_data, kde=True, color='blue', bins=30)

    # Q-Q Plot
    stats.probplot(residuals_data, dist="norm", plot=plt)
```

**Plots Generated:**
- **Histogram of Residuals** — checks if the residual distribution is bell-shaped.
- **Q-Q Plot (Quantile-Quantile)** — compares residual quantiles against a theoretical normal distribution. Points lying on the diagonal line confirm normality.

**Interpretation:** Deviations in the tails of the Q-Q plot indicate slight non-normality at extremes, a common pattern with real-world price data due to right-skewed targets.

---

### 2. No Multicollinearity

**Why it matters:** High multicollinearity inflates coefficient variance, making individual predictor estimates unstable and unreliable.

**How it was checked — Variance Inflation Factor (VIF):**

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(x):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = x.columns
    vif_data['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif_data
```

**VIF Results:**

| Feature | VIF | Interpretation |
|---|---|---|
| `transmission_Manual` | **9.74** | High — approaching problematic multicollinearity |
| `Age` | 7.02 |  Moderate — warrants further investigation |
| `fuel_Petrol` | 6.84 |  Moderate |
| `fuel_Diesel` | 6.56 |  Moderate |
| `seller_type_Individual` | 4.81 | Acceptable |
| `km_driven` | 4.44 |  Acceptable |
| Remaining features | < 2 |  Little to no multicollinearity |

> **Rule of Thumb:** VIF > 10 → severe multicollinearity; VIF 5–10 → moderate; VIF < 5 → acceptable.

`transmission_Manual` and fuel dummies (Petrol, Diesel) are correlated with each other — expected since certain fuel types are predominantly paired with manual transmissions in the Indian used car market.

---

### 3. Homoscedasticity

**Why it matters:** OLS assumes **constant variance** of residuals across all predicted values. Violation (heteroscedasticity) leads to inefficient estimates and invalid significance tests.

**How it was checked:**

```python
def homoscedasticity_assumption(model, x_test, y_test):
    df_results = residuals(model, x_test, y_test)
    sns.regplot(
        x='Predicted', y='Residuals', data=df_results,
        lowess=True,
        color='#0055ff',
        line_kws={'color': '#ff7000', 'ls': '--', 'lw': 2.5}
    )
    plt.axhline(y=0, color='#23bf00', lw=1)
```

**Plot:** Residuals vs. Predicted Values with a LOWESS (Locally Weighted Scatterplot Smoothing) trendline.

**Interpretation:**
- The **orange trendline should remain flat** (≈ 0) across all predicted values to confirm homoscedasticity.
- A funnel-shaped spread indicates **heteroscedasticity** — a common issue when the target (`selling_price`) has a large range and skewed distribution.

---

## 📐 Ridge Regression (L2 Regularization)

Ridge Regression adds an **L2 penalty** (`λ · Σwᵢ²`) to the OLS cost function to shrink large coefficients and reduce overfitting — particularly useful when multicollinearity is present.

**Objective Function:**
```
J(w) = MSE + λ · Σwᵢ²
```

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=10)   # alpha = λ (regularization strength)
ridge_reg.fit(x_train_scaled, y_train)
```

**Key Behavior:**
- **Shrinks all coefficients** toward zero but never sets them exactly to zero.
- All features are **retained** in the model.
- Higher `alpha` → stronger regularization → more shrinkage.

**Evaluation:**
```python
ridge_train_mse = metrics.mean_squared_error(y_train, ridge_reg.predict(x_train_scaled))
ridge_test_mse  = metrics.mean_squared_error(y_test,  ridge_reg.predict(x_test_scaled))
```

The coefficient DataFrame was inspected to compare Ridge-shrunken weights against the baseline Linear Regression coefficients.

---

## 🎯 Lasso Regression (L1 Regularization)

Lasso Regression adds an **L1 penalty** (`λ · Σ|wᵢ|`) to the cost function — uniquely capable of driving irrelevant feature coefficients to **exactly zero**, performing automatic feature selection.

**Objective Function:**
```
J(w) = MSE + λ · Σ|wᵢ|
```

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=10000)  # Higher alpha for aggressive feature selection
lasso_reg.fit(x_train_scaled, y_train)
```

**Key Behavior:**
- Produces **sparse solutions** — drives some coefficients to exactly 0.
- Acts as a built-in **feature selector**.
- Higher `alpha` → more features zeroed out.

**Retained Features Inspection:**
```python
lasso_coefficients = lasso_reg.coef_
retained_features  = np.where(lasso_coefficients != 0)[0]
print("Lasso Retained Features (Indices):", retained_features)
```

**Evaluation:**
```python
lasso_train_mse = metrics.mean_squared_error(y_train, lasso_reg.predict(x_train_scaled))
lasso_test_mse  = metrics.mean_squared_error(y_test,  lasso_reg.predict(x_test_scaled))
```

---

## 📊 Model Comparison

| Model | Regularization | Key Property |
|---|---|---|
| **Linear Regression** | None | Baseline OLS — minimizes MSE directly |
| **Gradient Descent** | None | Custom NumPy implementation — same OLS objective, iterative solver |
| **Ridge Regression** | L2 (`α=10`) | Shrinks all coefficients; retains all features |
| **Lasso Regression** | L1 (`α=10,000`) | Zeros out irrelevant features; performs feature selection |

> Ridge is preferred when all features are potentially relevant but multicollinearity is present. Lasso is preferred when you suspect many features are irrelevant and want a simpler, interpretable model.

---

## 💡 Key Takeaways

- **Feature Engineering**: Converting `year` → `Age` made the relationship with `selling_price` more intuitive and linear.
- **High-Cardinality Drops**: Removing `name` (1,491 unique values) prevented overfitting and noise injection.
- **Gradient Descent from Scratch**: Confirms understanding of the mathematical core of all gradient-based ML optimizers — `w = w − α · ∇J(w)`.
- **VIF Analysis**: Revealed that `transmission_Manual` (VIF 9.74) and fuel dummies have moderate-to-high multicollinearity — Ridge Regression is especially well-suited for this dataset.
- **Lasso as Feature Selector**: With `alpha=10,000`, Lasso effectively identifies the most predictive subset of features, improving model parsimony.
- **Homoscedasticity Violation**: The residual fan-out pattern is common with skewed price targets — applying a log transformation on `selling_price` is a recommended next step.

---

## 📁 Repository Structure

```
├── Gradient_Descent.ipynb   # Main notebook
├── README.md                # Project documentation
└── data/
    └── CAR DETAILS FROM CAR DEKHO.csv
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels

# 3. Launch the notebook
jupyter notebook Gradient_Descent.ipynb
```
