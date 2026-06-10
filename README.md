# Gradient Descent — Car Price Prediction

A from-scratch implementation of the **Gradient Descent** optimisation algorithm applied to predict used car selling prices. The project walks through the full ML pipeline: exploratory data analysis, feature engineering, standard scaling, a scikit-learn Linear Regression baseline, and a hand-coded Gradient Descent solver — all on the [Car Dekho](https://www.cardekho.com/) dataset.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [What is Gradient Descent?](#what-is-gradient-descent)
  - [The Cost Function](#the-cost-function)
  - [The Update Rule](#the-update-rule)
  - [Learning Rate](#learning-rate)
  - [Epochs](#epochs)
  - [Types of Gradient Descent](#types-of-gradient-descent)
- [Project Workflow](#project-workflow)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)

---

## Problem Statement

Given a set of features about a used car — its age, fuel type, seller type, transmission, kilometres driven, and ownership history — **predict its selling price**.

---

## Dataset

| Property | Value |
|---|---|
| Source | Car Dekho |
| File | `CAR_DETAILS_FROM_CAR_DEKHO.csv` |
| Rows | 4,340 |
| Original Features | 8 |

**Raw columns:**

| Column | Type | Description |
|---|---|---|
| `name` | object | Car model name (1,491 unique values — dropped) |
| `year` | int | Manufacturing year → converted to `Age` |
| `selling_price` | int | Target variable (₹) |
| `km_driven` | int | Total kilometres driven |
| `fuel` | object | Fuel type (Petrol, Diesel, CNG, LPG, Electric) |
| `seller_type` | object | Individual / Dealer / Trustmark Dealer |
| `transmission` | object | Manual / Automatic |
| `owner` | object | First / Second / Third / Fourth & Above / Test Drive Car |

---

## What is Gradient Descent?

Gradient Descent is an iterative first-order optimisation algorithm used to minimise a differentiable function — in machine learning, that function is the **cost (loss) function** that measures how far the model's predictions are from the true values.

The algorithm takes small, repeated steps in the direction that most steeply *decreases* the cost, updating the model parameters at every step until it converges to a minimum.

### The Cost Function

For linear regression we use Mean Squared Error (MSE):

```
J(w) = (1 / 2m) * Σ (ŷᵢ − yᵢ)²
```

where:

- `m` — number of training samples
- `ŷᵢ = X · w` — predicted value for sample *i*
- `yᵢ` — actual target value for sample *i*
- `w` — weight (parameter) vector

The factor `1/2` is a convenience that cancels the `2` produced when differentiating, giving cleaner gradient expressions.

### The Update Rule

At each epoch, every weight is updated simultaneously:

```
w := w − α · ∇J(w)
```

The gradient with respect to the weight vector is:

```
∇J(w) = (1/m) · Xᵀ · (Xw − y)
```

Written out in full:

```
w := w − (α / m) · Xᵀ · (ŷ − y)
```

| Symbol | Meaning |
|---|---|
| `α` | Learning rate (step size) |
| `∇J(w)` | Gradient of the cost w.r.t. weights |
| `Xᵀ` | Transpose of the feature matrix |
| `ŷ − y` | Residual (error) vector |

### Learning Rate

The learning rate `α` controls how large each step is:

- **Too large** → the algorithm overshoots the minimum and may diverge.
- **Too small** → convergence is correct but extremely slow.
- **Just right** → cost decreases smoothly and reaches a minimum efficiently.

This project uses `α = 0.01`, which produces stable, monotonically decreasing cost curves.

### Epochs

One **epoch** is a single full pass through the training data. With each epoch the weights are refined and the cost decreases. This project trains for **1,000 epochs**, printing the cost every 100 epochs so convergence can be monitored.

### Types of Gradient Descent

| Variant | Update frequency | Pros | Cons |
|---|---|---|---|
| **Batch GD** *(used here)* | Once per epoch (all samples) | Stable convergence, exact gradient | Slow on large datasets |
| **Stochastic GD (SGD)** | Once per sample | Fast updates, escapes local minima | Noisy; high variance |
| **Mini-batch GD** | Once per mini-batch | Balance of speed and stability | Requires tuning batch size |

---

## Project Workflow

```
Raw CSV
   │
   ▼
Data Preparation
  • Drop 'name' (1,491 unique values)
  • Replace 'year' with 'Age' (max_year + 1 − year)
  • Check nulls & duplicates
   │
   ▼
Exploratory Data Analysis
  • Count plots for categorical features
  • Box plots: categorical vs. selling price
  • Histograms & KDE plots for numeric features
  • Scatter plots: Age & km_driven vs. selling price
   │
   ▼
Feature Engineering
  • One-Hot Encoding on [fuel, seller_type, transmission, owner]
  • Correlation heatmap
   │
   ▼
Train / Test Split  (80 / 20, random_state = 42)
   │
   ▼
Standard Scaling  (fit on train, transform both)
   │
   ├──► Scikit-learn Linear Regression (baseline)
   │         Evaluate: MAE, MSE, RMSE, R²
   │
   └──► Custom Gradient Descent
             • Prepend bias column of ones
             • 1,000 epochs, α = 0.01
             • Plot cost curve
```

---

## Implementation Details

The core implementation lives in the `gradient_descent` function:

```python
def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    m = x.shape[0]          # number of samples
    n = x.shape[1]          # number of features
    weights = np.zeros(n)   # initialise weights to zero
    cost_history = []

    for epoch in range(epochs):
        predictions  = np.dot(x, weights)              # ŷ = Xw
        errors       = predictions - y                 # residuals
        gradients    = (1 / m) * np.dot(x.T, errors)  # ∇J(w)
        weights     -= learning_rate * gradients       # weight update
        cost         = (1 / (2 * m)) * np.sum(errors ** 2)  # MSE
        cost_history.append(cost)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")

    return weights, cost_history
```

**Key design decisions:**

- **Zero initialisation** — weights start at zero; because the data is scaled, this is a stable and neutral starting point.
- **Bias term** — a column of ones is prepended to `X` before training (`np.hstack`) so the intercept is learned as `weights[0]` without special-casing.
- **Vectorised numpy operations** — all matrix multiplications use `np.dot`, avoiding slow Python loops over samples.
- **Cost history logging** — every epoch's cost is appended to `cost_history`, enabling the convergence plot.

---

## Results

After training, the cost curve is visualised to confirm that Gradient Descent converged:

```python
plt.plot(range(len(cost_history)), cost_history)
plt.title("Cost Function over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.show()
```

A steadily decreasing curve indicates that the weights are converging toward the optimal solution. The scikit-learn `LinearRegression` baseline (evaluated with MAE, MSE, RMSE, and R²) provides a reference against which the custom implementation can be compared.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook

```bash
jupyter notebook "Gradient_Descent.ipynb"
```

Ensure `CAR_DETAILS_FROM_CAR_DEKHO.csv` is in the same directory as the notebook, then run all cells from top to bottom.

---

## Project Structure

```
.
├── Gradient_Descent.ipynb          # Main notebook
├── CAR_DETAILS_FROM_CAR_DEKHO.csv  # Dataset
└── README.md                       # This file
```
