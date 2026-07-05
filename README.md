# Formative 3 — Group 6

Implementation of the EM algorithm, Bayesian probability, and gradient descent, submitted for Formative Assessment 3.

## Team Members & Contributions

| Member | Part(s) Contributed |
|---|---|
| Yusuf Nabide | Part 3 — Manual Gradient Descent (Iteration 1) |
| Davy Mugire | Part 3 — Manual Gradient Descent (Iteration 2) |
| Faith Irakoze | Part 3 — Manual Gradient Descent (Iteration 3) |
| Sedem Amuzu | Part 3 — Manual Gradient Descent (Iteration 4) |

*(Add each member's contribution to Parts 1, 2, and 4 here as well, so every member's involvement across all parts is visible.)*

## Repository Structure

```
├── GaltonFamilies.csv                       # Dataset used for Part 1 (parent/child heights)
├── Part_1_Gaussian_Distributions.ipynb      # Part 1: EM algorithm for Gaussian Mixture Model
├── Part 2 - Bayesian Probability.ipynb      # Part 2: Bayes' Theorem on IMDb movie reviews
├── Group 6 _ Manual Calculation -Part 3.pdf # Part 3: Handwritten manual gradient descent (4 iterations)
├── Group6_Part4.ipynb                       # Part 4: Gradient descent implemented in Python/SciPy
└── README.md
```

## Part 1: EM Algorithm for Gaussian Mixture Models

**File:** `Part_1_Gaussian_Distributions.ipynb`
**Dataset:** `GaltonFamilies.csv`

Implements the Expectation-Maximization algorithm from scratch to separate a mixture of two Gaussian distributions (parent heights vs. child heights) without using population labels.

- Initializes μ, σ², and mixing coefficients (π) for both components
- **E-step:** computes soft responsibilities (posterior probability of cluster membership) for each data point
- **M-step:** updates μ, σ², and π using the responsibilities
- Tracks log-likelihood across iterations to monitor convergence
- Includes classification demo: given a new height, outputs posterior probability of belonging to each group
- Discusses why splitting at the global mean (hard assignment) is a weaker approach than soft probabilistic assignment via EM

To run:
```bash
jupyter notebook Part_1_Gaussian_Distributions.ipynb
```

## Part 2: Bayesian Probability on Movie Reviews

**File:** `Part 2 - Bayesian Probability.ipynb`

Applies Bayes' Theorem to sentiment keywords from the IMDb Movie Reviews dataset, implemented using basic Python only (no ML libraries).

- Selected keywords indicating positive and negative sentiment
- Computes for each keyword: Prior P(Positive), Likelihood P(keyword\|Positive), Marginal P(keyword), Posterior P(Positive\|keyword)
- All Bayes' Theorem logic implemented manually in Python

To run:
```bash
jupyter notebook "Part 2 - Bayesian Probability.ipynb"
```

## Part 3: Manual Gradient Descent (Handwritten)

**File:** `Group 6 _ Manual Calculation -Part 3.pdf`

Manual computation of 4 gradient descent updates for a linear regression model `y = m1*x1 + m2*x2 + b`, using matrix multiplication throughout (no scalar shortcuts).

- Derives ∂E/∂m and ∂E/∂b explicitly via the chain rule
- Computes predictions, error, and MSE at each step
- 4 iterations performed, one per group member, learning rate α = 0.01
- Shows the trend in m, b, and error across iterations

| Iteration | m | b | MSE |
|---|---|---|---|
| 0 (init) | [-1, 2] | [1, 1] | 61.0 |
| 1 | [-1.45, 0.87] | [0.99, 0.89] | 6.5 |
| 2 | [-1.33, 1.18] | [1.019, 0.91] | 2.53 |
| 3 | [-1.363, 1.121] | [1.057, 0.887] | 2.17 |

## Part 4: Gradient Descent in Code

**File:** `Group6_Part4.ipynb`

Converts the Part 3 manual calculations into Python, using SciPy to compute derivatives and validate the manual gradient derivation.

- Function to compute the derivative of the cost function using SciPy
- Iterative update of m and b, with every calculation step visible (not abstracted away)
- Final predictions computed using the converged m and b
- Matplotlib plots:
  - m and b values over iterations
  - Error (MSE) over iterations

To run:
```bash
jupyter notebook Group6_Part4.ipynb
```

## Requirements

```bash
pip install numpy scipy matplotlib pandas jupyter
```

## Notes

- Part 1 uses `GaltonFamilies.csv` as the height dataset (parent vs. child heights), treated as an unlabeled two-component Gaussian mixture.
- Part 2 uses only base Python for probability calculations — no external ML libraries.
- Part 3 is submitted as a handwritten, scanned PDF per assignment requirements.
- Part 4 mirrors Part 3's manual math exactly, so results can be cross-checked between the two.
