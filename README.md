# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student Name:** Puneet Madan  
**Roll Number:** U20230120  
**GitHub Branch:** `puneet_U20230120`

---

## Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** system for recommending news articles. The system learns which news category maximises user engagement for each user type, using three bandit algorithms: Epsilon-Greedy, Upper Confidence Bound (UCB), and SoftMax. A supervised classifier first identifies the user type (context), and the trained bandit policy then selects the optimal news category (arm).

---

## Approach and Design Decisions

### 1. Data Preprocessing

The dataset consists of 209,527 news articles and 2,000 train / 2,000 test user records.

**Missing values:** Numerical columns are filled with the column median; categorical columns use the mode. The `age` column had the most missingness (~35% of user records).

**Feature encoding:** Three categorical user features (`user_id`, `browser_version`, `region_code`) are label-encoded. A `StandardScaler` is applied before classification.

**News category mapping:** 42 original HuffPost categories were collapsed into 4 bandit arms using domain-knowledge grouping:

| Bandit Arm | Mapped from | Article count |
|---|---|---|
| Crime | POLITICS, CRIME, U.S. NEWS | 40,541 |
| Education | WELLNESS, PARENTING, EDUCATION | 27,750 |
| Entertainment | ENTERTAINMENT, COMEDY | 22,762 |
| Tech | TECH, SCIENCE, BUSINESS | 10,302 |

Crime and Education have the largest article pools, ensuring recommendation diversity.

---

### 2. User Classifier (Context Detector)

**Model:** XGBoost (`n_estimators=200`, `learning_rate=0.05`, `max_depth=7`, `eval_metric=mlogloss`)  
**Train/validation split:** 80% / 20% from `train_users.csv` (1,600 / 400 samples), stratified.

**Validation results:**

```
              precision    recall  f1-score   support
      user_1       0.88      0.86      0.87       142
      user_2       0.99      0.87      0.93       142
      user_3       0.83      0.98      0.90       116
    accuracy                           0.90       400
   macro avg       0.90      0.91      0.90       400
weighted avg       0.91      0.90      0.90       400

Confusion Matrix:
[[122   1  19]
 [ 14 124   4]
 [  2   0 114]]
```

**Overall validation accuracy: 90.0%**. The model is strongest at identifying user_2 (99% precision) and weakest at separating user_1 from user_3 (19 user_1 samples misclassified as user_3).

**Predictions on test_users.csv (n=2,000):**

| Predicted Label | Count | % |
|---|---|---|
| user_1 | 763 | 38.15% |
| user_2 | 705 | 35.25% |
| user_3 | 532 | 26.60% |

This distribution is consistent with the training label distribution (user_2: 35.6%, user_1: 35.35%, user_3: 29.05%), indicating the classifier generalises well.

---

### 3. Contextual Bandit Setup

**Arm index mapping (from the lab specification):**

| j | News Category | User Context |
|---|---|---|
| 0 | Entertainment | user_1 |
| 1 | Education | user_1 |
| 2 | Tech | user_1 |
| 3 | Crime | user_1 |
| 4 | Entertainment | user_2 |
| 5 | Education | user_2 |
| 6 | Tech | user_2 |
| 7 | Crime | user_2 |
| 8 | Entertainment | user_3 |
| 9 | Education | user_3 |
| 10 | Tech | user_3 |
| 11 | Crime | user_3 |

Each algorithm maintains a separate Q-value table `Q[context][arm]` with incremental mean updating. The sampler was initialised with `ROLL_NUMBER = 120`.

---

### 4. Bandit Algorithms Implemented

All three algorithms were run for **T = 10,000 steps** with contexts sampled uniformly at random each step.

#### 4.1 Epsilon-Greedy

With probability ε, select a random arm (explore); otherwise select `argmax Q[ctx]` (exploit).

**Hyperparameter sweep results:**

| ε | Mean Reward (T=10,000) | Δ vs best |
|---|---|---|
| **0.01** | **5.3699** | — |
| 0.10 | 4.8464 | −9.8% |
| 0.30 | 3.9321 | −26.8% |

**Best ε = 0.01.** Final simulation mean reward: **5.3310**.

**Learned Q-table (ε=0.01):**

| | Entertainment | Education | Tech | Crime |
|---|---|---|---|---|
| **user_1** | **9.4321** | 0.4516 | −2.2948 | −7.2864 |
| **user_2** | 0.0346 | −5.7825 | 0.1301 | **0.0683** |
| **user_3** | 5.3252 | **6.7123** | 1.8917 | −6.9647 |

#### 4.2 UCB (Upper Confidence Bound)

UCB score = `Q[ctx][arm] + C × √(ln(t+1) / (count[ctx][arm] + ε))`. Selects the arm with the highest upper confidence bound.

**Hyperparameter sweep results:**

| C | Mean Reward (T=10,000) | Δ vs best |
|---|---|---|
| **0.5** | **5.4410** | — |
| 2.0 | 5.3651 | −1.4% |
| 1.0 | 5.3488 | −1.7% |

**Best C = 0.5.** Final simulation mean reward: **5.3866** *(highest of all three algorithms)*.

**Learned Q-table (C=0.5):**

| | Entertainment | Education | Tech | Crime |
|---|---|---|---|---|
| **user_1** | **9.4545** | −1.1061 | −2.1682 | −5.9653 |
| **user_2** | −0.8028 | −4.8927 | −0.3807 | **0.1366** |
| **user_3** | 5.2536 | **6.7076** | 1.2770 | −6.1619 |

#### 4.3 SoftMax (τ = 1, fixed)

Arm selection probability ∝ `exp(Q[ctx][arm] / τ)`. At τ=1, selection is proportional to the exponentiated Q-values.

**Final mean reward: 5.2224** *(lowest of the three)*.

**Learned Q-table (τ=1):**

| | Entertainment | Education | Tech | Crime |
|---|---|---|---|---|
| **user_1** | **9.4406** | −1.1495 | 0.0000 | −9.8404 |
| **user_2** | −0.3062 | −5.6408 | 0.1594 | **0.0953** |
| **user_3** | 5.4057 | **6.7349** | 1.4204 | −7.0354 |

---

## Results and Analysis

### Overall Performance Comparison

| Algorithm | Best Hyperparameter | Mean Reward (sweep) | Mean Reward (final sim) | Rank |
|---|---|---|---|---|
| UCB | C = 0.5 | 5.4410 | **5.3866** | 🥇 1st |
| Epsilon-Greedy | ε = 0.01 | 5.3699 | 5.3310 | 🥈 2nd |
| SoftMax | τ = 1.0 (fixed) | 5.3356 | 5.2224 | 🥉 3rd |

UCB achieves the best performance overall. The margin between UCB and Epsilon-Greedy is small (~1.1%), but the margin to SoftMax is more notable (~7.0% gap from UCB).

---

### Hyperparameter Sensitivity Analysis

**Epsilon-Greedy is highly sensitive to ε:**  
The reward drops sharply and monotonically from 5.3699 (ε=0.01) to 3.9321 (ε=0.30) — a 26.8% degradation. This is because the arm reward gaps in this environment are very large (e.g. user_1: Entertainment ≈ +9.4 vs Crime ≈ −7.3, a spread of ~16.7 reward units). Any significant random exploration wastes steps on deeply negative arms. A fixed ε does not decay over time, so even at T=10,000, ε=0.30 still wastes 30% of pulls on random arms.

**UCB is robust to C:** All three C values achieve within 1.7% of each other (5.3488–5.4410). The confidence bound mechanism naturally scales exploration with arm uncertainty, meaning the exact value of C matters less once arms have been sampled sufficiently. C=0.5 is optimal here — enough to explore uncertain arms early without over-exploring once Q-values have stabilised.

---

### Context-Specific Analysis

**user_1 — Clear dominant arm: Entertainment (~9.44)**  
This is the easiest context to learn. The reward landscape is highly differentiated: Entertainment is strongly positive, Education mildly positive, Tech negative, Crime strongly negative (≈ −7 to −9.8). All three algorithms identify Entertainment quickly. The severe penalty for Crime makes poor exploration choices especially costly.

**user_2 — Weak signal, flat reward landscape**  
All four arms produce near-zero rewards (range ≈ −5.8 to +0.14). Crime is marginally best (~0.07–0.14), but the signal is so weak that no algorithm confidently converges. This context contributes the most uncertainty and drag on mean reward. The flat landscape is a genuine challenge for any bandit algorithm.

**user_3 — Two competitive arms: Education (~6.71) and Entertainment (~5.26–5.41)**  
This is the most interesting context. Both arms are strongly positive, but Education leads by ~1.3–1.5 reward units. UCB handles this best because its uncertainty-driven exploration ensures both arms are sampled sufficiently to distinguish them. SoftMax also samples both frequently due to its probabilistic selection. Crime is again strongly negative (≈ −6 to −7).

**Cross-context pattern:** Crime is a poor recommendation across all contexts — negative for user_1 (−7.3) and user_3 (−6.5), and only marginally positive for user_2 (+0.07). Tech and Education show strong context dependence (excellent for user_3, poor for user_2).

---

### Strengths and Limitations

| Algorithm | Strengths | Limitations |
|---|---|---|
| **Epsilon-Greedy** (ε=0.01) | Simple, interpretable. Fast convergence when arm gaps are large and ε is well-tuned. Minimal computational overhead. | Highly sensitive to ε. Fixed ε never decays — wasteful exploration at step 9,999 same as step 1. Performs poorly with high ε values. Cannot adapt to non-stationary rewards without decay. |
| **UCB** (C=0.5) | Best overall performance. Principled exploration: arms are revisited in proportion to uncertainty. Robust to hyperparameter choice (only 1.7% spread across C values). No need for random exploration. | Slightly more complex. Global t-counter shared across contexts can over-explore early. Assumes stationary reward distributions. |
| **SoftMax** (τ=1) | Soft probabilistic selection avoids hard exploitation. Smooth convergence curve. Never completely ignores any arm — useful when reward distributions are uncertain or non-stationary. | Fixed τ=1 gives non-negligible probability to clearly bad arms (e.g. Crime at −9.8 for user_1), dragging down mean reward. Without τ-decay, exploration persists throughout the horizon regardless of confidence. Lowest performance of the three. |

---

### Recommendation Engine

The end-to-end pipeline:
1. **Classify** → XGBoost predicts user type from 32 input features
2. **Select category** → Bandit policy for that context selects the optimal arm
3. **Recommend** → Random article sampled from the selected category pool

Sample outputs (5 test users):

| User | Predicted Context | EG Recommends | UCB Recommends | SoftMax Recommends |
|---|---|---|---|---|
| 1 | user_2 | Tech | Crime | Tech |
| 2 | user_1 | Entertainment | Entertainment | Entertainment |
| 3 | user_1 | Entertainment | Entertainment | Entertainment |
| 4 | user_1 | Entertainment | Entertainment | Entertainment |
| 5 | user_1 | Entertainment | Entertainment | Entertainment |

The dominance of Entertainment for user_1 is clearly captured. SoftMax occasionally selects Tech for user_2 due to its probabilistic arm selection, which reflects the flat and uncertain reward landscape for that context.

---

## Reproducing the Experiments

### Requirements

```
python >= 3.12
numpy, pandas, matplotlib
scikit-learn
xgboost
lightgbm
rlcmab-sampler
```

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm
pip install rlcmab-sampler
```

### Data

Place the following files in a `data/` directory at the repository root:

```
data/
├── news_articles.csv
├── train_users.csv
└── test_users.csv
```

### Running

```bash
jupyter notebook lab3_results_U20230120.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).

> **Important:** Set `ROLL_NUMBER` in the sampler initialisation cell to your own roll number before running.

---

## Repository Structure

```
├── lab3_results_U20230120.ipynb   # Main notebook — all code, outputs, and plots
├── README.md                      # This file — project report and analysis
└── data/                          # (not committed) — place datasets here
    ├── news_articles.csv
    ├── train_users.csv
    └── test_users.csv
```

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 2: Multi-armed Bandits.
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. *Machine Learning*, 47(2–3), 235–256.
- Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis*. Wiley. (SoftMax/Boltzmann exploration.)
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
