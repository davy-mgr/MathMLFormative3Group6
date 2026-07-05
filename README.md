# Formative 3 — Group 6

Implementation of the EM algorithm, Bayesian probability, and gradient descent, submitted for Formative Assessment 3.

## Team Members

| Member |
|---|
| Yusuf Nabide |
| Davy Mugire |
| Faith Irakoze |
| Sedem Amuzu |

## Repository Structure

```
├── GaltonFamilies.csv
├── Part_1_Gaussian_Distributions.ipynb
├── Part 2 - Bayesian Probability.ipynb
├── Group 6 _ Manual Calculation -Part 3.pdf
├── Group6_Part4.ipynb
└── README.md
```

## Part 1: EM Algorithm for Gaussian Mixture Models

**File:** `Part_1_Gaussian_Distributions.ipynb`
**Dataset:** `GaltonFamilies.csv`

Implement the Expectation-Maximization algorithm from scratch to separate a mixture of two Gaussian distributions (parent heights vs. child heights) without using population labels.

- Initialize μ, σ², and mixing coefficients (π) for both components
- **E-step:** compute soft responsibilities for each data point
- **M-step:** update μ, σ², and π using the responsibilities
- Track log-likelihood across iterations to monitor convergence
- Includes classification demo: given a new height, output posterior probability of belonging to each group
- Splitting at the global mean is a weaker approach than soft probabilistic assignment via EM


## Part 2: Bayesian Probability on Movie Reviews

**File:** `Part 2 - Bayesian Probability.ipynb`

Apply Bayes' Theorem to sentiment keywords from the IMDb Movie Reviews dataset.

- Select three keywords indicating positive and negative sentiment each.
- Compute for each keyword: Prior P(Positive), Likelihood P(keyword\|Positive), Marginal P(keyword), Posterior P(Positive\|keyword)
- All Bayes' Theorem logic implemented manually in Python

## Part 3: Manual Gradient Descent (Handwritten)

**File:** `Group 6 _ Manual Calculation -Part 3.pdf`

Manual computation of 4 gradient descent updates for a linear regression model `y = m1*x1 + m2*x2 + b`.

- Derive ∂E/∂m and ∂E/∂b explicitly via the chain rule
- Compute predictions, error, and MSE at each step
- Perform 4 iterations with a learning rate α = 0.01
- The error is reducing each iteration but the error reduces by a less margin every time. 

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
