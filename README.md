# 🫀 Heart Disease Classification: A Systematic ML Comparison

---

## 📌 Project Overview

Cardiovascular disease is the **leading cause of global mortality**, responsible for approximately **17.9 million deaths annually** (WHO, 2024). A critical clinical challenge is that many patients deteriorate silently, physiological markers decline without triggering standard diagnostic flags, leaving preventable cardiac events undetected.

This project builds and systematically compares **six machine learning classifiers** for **binary heart disease prediction**, using a single consolidated clinical dataset drawn from five independent cardiovascular databases. The goal: determine whether algorithmic complexity actually drives predictive performance on structured clinical tabular data or whether **feature quality is the real ceiling**.

---

## 🗃️ Dataset

| Property | Detail |
|---|---|
| **Source** | [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) — Fedesoriano (2021) |
| **Records** | 918 raw → **917 after cleaning** (1 removed: RestingBP = 0, physiologically impossible) |
| **Features** | 11 clinical input features → **15 after one-hot encoding** |
| **Target** | `HeartDisease` - binary (1 = disease, 0 = no disease) |
| **Class Balance** | 507 positive (55.3%) · 410 negative (44.7%) - mild 55:45 imbalance |
| **Source Databases** | Cleveland · Hungary · Switzerland · Long Beach VA · Statlog |

### 🔬 Clinical Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age in years |
| Sex | Categorical | Biological sex (M/F) |
| ChestPainType | Categorical | ASY · ATA · NAP · TA |
| RestingBP | Numeric | Resting blood pressure (mmHg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dL) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dL |
| RestingECG | Categorical | ECG result (Normal · ST · LVH) |
| MaxHR | Numeric | Max heart rate during exercise |
| ExerciseAngina | Categorical | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression (exercise vs rest) |
| STSlope | Categorical | Peak exercise ST slope (Up · Flat · Down) |

---

## 📊 Exploratory Data Analysis- Key Findings

### Age Group Risk Profile

| Age Group | Disease Rate | Patient Count |
|---|---|---|
| Under 40 | 32.5% | 80 |
| 40–49 | 40.3% | 211 |
| 50–59 | 56.6% | 373 |
| 60+ | **73.1%** | 253 |

> ⚠️ Even the youngest cohort (under 40) carries a **32.5% disease rate** - reinforcing that clinical monitoring cannot be deferred to late middle age.

### Top Predictors (Point-Biserial Correlation with HeartDisease)

| Feature | Correlation (r) | Direction | Significant? |
|---|---|---|---|
| Oldpeak | 0.404 | Positive | ✅ Yes |
| MaxHR | –0.401 | Negative | ✅ Yes |
| Age | 0.282 | Positive | ✅ Yes |
| RestingBP | 0.118 | Positive | ✅ Yes |
| Cholesterol | 0.095 | Positive | ✅ Yes |

### STSlope × Oldpeak Interaction

The interaction between `STSlope` and `Oldpeak` produces a **near-definitive disease signature**:

| STSlope | Healthy (Mean Oldpeak) | Disease+ (Mean Oldpeak) |
|---|---|---|
| Up | 0.24 | 0.59 |
| Flat | 0.55 | **1.57** |
| Down | 0.89 | **2.37** |

This is why tree-based models excel - they capture **conditional feature importance** that a linear boundary cannot express.

---

## 🔧 Preprocessing Pipeline

### 1. Data Cleaning
- Removed **1 record** with `RestingBP = 0` (physiologically impossible) → 917 records retained

### 2. Cholesterol Imputation - KNN vs 5 Alternatives

171 zero-coded Cholesterol values (18.6% of the dataset) required imputation. Six strategies were benchmarked under 10-fold CV using a Random Forest:

| Method | Accuracy | F1-Score | Precision | Recall | Fair? |
|---|---|---|---|---|---|
| Global Median | 0.8570 | 0.8741 | 0.8583 | 0.8935 | ✅ Yes |
| Stratified (Age+Sex) | 0.8592 | 0.8755 | 0.8649 | 0.8896 | ✅ Yes |
| **KNN (no target) ✅** | **0.8614** | **0.8771** | **0.8664** | **0.8916** | ✅ Yes |
| KNN (with target) ⚠️ | 0.8636 | 0.8781 | 0.8715 | 0.8876 | ❌ Leakage |
| MICE (no target) | 0.8560 | 0.8728 | 0.8599 | 0.8896 | ✅ Yes |
| Drop Rows | 0.8632 | 0.8590 | 0.8469 | 0.8737 | ✅ Yes (n=746) |

**Selected: KNN (k=5), target excluded.** Reasons:
- ✅ Best F1 among clinically valid (fair) methods
- ✅ Preserves ~85% of original Cholesterol-HeartDisease class gap
- ✅ Generates **139 unique personalised values** (not one flat estimate)
- ✅ Mirrors real clinical reasoning - find 5 most similar patients by features, average their cholesterol
- ✅ No data leakage - target excluded from neighbour search

> Post-imputation: Mean = 245.0 mg/dL, SD = 54.7 (vs. pre-imputation: Mean = 244.6, SD = 59.2) - distribution shape preserved.

### 3. Outlier Retention

IQR screening flagged outliers but **all were retained** 0 extreme clinical readings (e.g., BP = 170, Cholesterol > 400) are statistically unusual but clinically meaningful in a cardiac population.

| Feature | Flagged Outliers |
|---|---|
| RestingBP | 27 |
| Cholesterol | 28 |
| Oldpeak | 16 |
| MaxHR | 2 |

### 4. Multicollinearity Check (VIF)

All numeric features returned VIF well below the threshold of 5 - no features dropped:

| Feature | VIF | Status |
|---|---|---|
| Age | 1.29 | ✅ No collinearity |
| MaxHR | 1.18 | ✅ No collinearity |
| RestingBP | 1.10 | ✅ No collinearity |
| Oldpeak | 1.09 | ✅ No collinearity |
| Cholesterol | 1.01 | ✅ No collinearity |

### 5. Encoding & Scaling
- **One-hot encoding** (`drop_first=True`) expanded features from **11 → 15**
- **StandardScaler** applied inside `sklearn.Pipeline` for SVM, Logistic Regression, and MLP - **fitted on training fold only** (no leakage)
- Tree-based models (Gradient Boosting, Random Forest) and Naive Bayes received **no scaling** (scale-invariant)

---

## 🤖 Models & Hyperparameter Tuning

Six algorithms selected to represent a **full spectrum of learning strategies**:

| Model | Family | Scaling Required |
|---|---|---|
| Gradient Boosting | Boosting Ensemble | ❌ No |
| Random Forest | Bagging Ensemble | ❌ No |
| Logistic Regression | Linear | ✅ Yes (Pipeline) |
| Support Vector Machine | Kernel-Based | ✅ Yes (Pipeline) |
| Naive Bayes | Probabilistic | ❌ No |
| MLP Neural Network | Neural Network | ✅ Yes (Pipeline) |

**Tuning method:** `GridSearchCV` with **5-fold inner CV**, scored on **F1**. Final evaluation on **10-fold stratified outer CV** (tuning and evaluation never share data).

### Tuned Hyperparameters

| Model | Key Tuned Parameters | Notable Change |
|---|---|---|
| Gradient Boosting | `learning_rate=0.05`, `n_estimators=100`, `max_depth=3` | Halved learning rate — prevents aggressive over-correction on 917 patients |
| Random Forest | `n_estimators=100`, `max_depth=15`, `min_samples_leaf=2` | Depth cap + leaf minimum — prevents memorisation |
| Logistic Regression | `C=0.01`, `solver=lbfgs` | Most aggressive regularisation — keeps coefficients stable |
| SVM | `C=1`, `kernel=linear`, `gamma=scale` | Switched RBF → Linear — flat hyperplane generalises equally well |
| MLP Neural Network | `hidden_layer_sizes=(128, 64)`, `activation=relu`, `alpha=0.0001` | Widened first layer (64→128) — more capacity for 15 features |
| Naive Bayes | `var_smoothing=1e-5` | 4 orders of magnitude above default — stabilises near-zero within-class variances |

---

## 📈 Results

### Baseline vs Tuned Performance

| Model | Baseline Acc | Tuned Acc | Baseline F1 | Tuned F1 | Δ F1 |
|---|---|---|---|---|---|
| Gradient Boosting | 0.8592 | 0.8658 | 0.8751 | **0.8815** | ↑ +0.0064 |
| Random Forest | 0.8570 | 0.8647 | 0.8739 | 0.8806 | ↑ +0.0067 |
| Logistic Regression | 0.8615 | 0.8636 | 0.8768 | 0.8795 | ↑ +0.0027 |
| SVM | 0.8603 | 0.8626 | 0.8782 | 0.8779 | → –0.0003 |
| Naive Bayes | 0.8549 | 0.8593 | 0.8685 | 0.8740 | ↑ +0.0055 |
| MLP Neural Network | 0.8441 | 0.8549 | 0.8609 | 0.8723 | ↑ +0.0114 |

### Final Model Rankings (Tuned, 10-Fold CV, n=917)

| Rank | Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| 🥇 1 | Gradient Boosting | 0.8658 | **0.8815** | **0.9262** |
| 🥈 2 | Random Forest | 0.8647 | 0.8806 | 0.9220 |
| 🥉 3 | Logistic Regression | 0.8636 | 0.8795 | 0.9222 |
| 4 | SVM | 0.8626 | 0.8779 | 0.9195 |
| 5 | Naive Bayes | 0.8593 | 0.8740 | 0.9209 |
| 6 | MLP Neural Network | 0.8549 | 0.8723 | 0.9144 |

> 📊 **Total F1 spread across all 6 models: 0.0092** (less than 1 percentage point)

### Overfitting Analysis

| Model | Train Accuracy | CV Accuracy | Gap | Status |
|---|---|---|---|---|
| Logistic Regression | 86.59% | 86.36% | **0.22%** | ✅ Good |
| SVM | 86.70% | 86.26% | **0.44%** | ✅ Good |
| Naive Bayes | 86.04% | 85.93% | **0.11%** | ✅ Good |
| Gradient Boosting | 91.38% | 86.58% | **4.80%** | ✅ Good |
| MLP Neural Network | 87.02% | 85.49% | **1.54%** | ✅ Good |
| Random Forest | 95.09% | 86.47% | **8.62%** | ⚠️ Mild Overfit |

### Statistical Significance Testing

Both **paired t-tests** and **McNemar's tests** were run across all 15 pairwise model comparisons on the 10-fold CV scores:

> **Result: No pairwise comparison reached p < 0.05.**

All performance differences are **statistically non-significant** — confirming that on structured clinical tabular data at this scale, **feature quality governs the performance ceiling more than algorithmic complexity.**

---

## 🔍 Feature Importance

Consistent top predictors across Gradient Boosting, Random Forest, and Logistic Regression:

| Feature | Clinical Interpretation |
|---|---|
| **STSlope** | Flat/Down slope = high disease probability |
| **Oldpeak** | ST depression > 0 = strong disease indicator |
| **ChestPainType** | ASY (asymptomatic) paradoxically = highest disease prevalence |
| **MaxHR** | Lower max heart rate at same age = disease signal |
| **ExerciseAngina** | Exercise-induced chest pain ≈ near-definitive risk marker |

---

## 🏆 Conclusion & Clinical Recommendation

- **Gradient Boosting** achieves the highest F1 (0.8815) and AUC (0.9262)
- The **entire performance spread** across six tuned models spans just **1.09 percentage points** in accuracy
- This convergence is statistically confirmed as **non-significant** by both t-tests and McNemar's tests

### ⭐ Recommended for Clinical Deployment: Logistic Regression

Despite Gradient Boosting's marginal numerical lead, **Logistic Regression is the most operationally compelling candidate**:

- Equivalent predictive performance (Acc: 0.8636, F1: 0.8795, AUC: 0.9222)
- **Full coefficient-level interpretability** — clinicians can understand *why* a patient is flagged
- No overfitting (train vs CV gap: 0.22%)
- Aligns with growing **regulatory and clinical governance requirements** for explainable medical AI

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

**Libraries:** `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scipy`

**Key techniques:** `GridSearchCV` · `StratifiedKFold` · `Pipeline` · `KNNImputer` · `StandardScaler` · `GradientBoostingClassifier` · `RandomForestClassifier` · `LogisticRegression` · `SVC` · `GaussianNB` · `MLPClassifier`

---

## 📬 Contact

**Aritra Paul** — Master of Business Analytics, Sunway University  
📧 aritrapaul30@gmail.com 
🔗 [GitHub](https://github.com/aritrapaul30)

---
