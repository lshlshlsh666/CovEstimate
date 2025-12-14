# Covariance Matrix Estimation for Financial Time Series

A comprehensive Python library for estimating covariance matrices in financial applications, implementing both unconditional and conditional covariance estimation methods. This project is part of the Baruch MFE Capstone.

## Features

### Unconditional Covariance Estimators

- **SampleCov**: Direct sample covariance matrix estimation
- **QIS (Quadratic Inverse Shrinkage)**: Advanced shrinkage estimator for high-dimensional settings
- **LS (Linear Shrinkage)**: Linear shrinkage estimator with tunable shrinkage parameter
- **AO (Average Oracle)**: Average Oracle method for covariance matrix estimation with multiprocessing support

### Conditional Covariance Estimators

- **RawEstimator**: Returns the unconditional covariance matrix without modification
- **DCCEstimator**: Dynamic Conditional Correlation (DCC) model for time-varying correlation estimation

### Devolatization

- **GARCH-based Devolatization**: Fit univariate GARCH models to estimate conditional volatilities for each asset

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

Or using traditional pip:

```bash
pip install -e .
```

## Requirements

- Python >= 3.11
- numpy >= 2.3.5
- pandas >= 2.3.3
- scipy >= 1.16.3
- matplotlib >= 3.10.8
- arch >= 8.0.0 (for GARCH models)
- tqdm >= 4.67.1
- jupyter >= 1.1.1

## Quick Start

### Basic Usage

```python
import numpy as np
from core.unconditional_estimator import SampleCov, QIS, LS, AO
from core.contional_estimator import DCCEstimator
from core.devolatizer import fit_garch_and_get_std

# Generate sample return data
T, n = 1000, 10  # 1000 days, 10 assets
R = np.random.randn(T, n)  # Return matrix [T, n]

# Estimate volatility using GARCH
D = fit_garch_and_get_std(R)  # Volatility matrix [T, n]

# Unconditional covariance estimation
sample_cov = SampleCov(R, D)
C_sample = sample_cov.estimate()

qis = QIS(R, D)
C_qis = qis.estimate()

ls = LS(R, D, rho=0.3)
C_ls = ls.estimate()

ao = AO(R, D, lookback_window=200)
ao.fit(n_jobs=4)  # Use 4 parallel processes
C_ao = ao.estimate()

# Conditional covariance estimation (DCC)
dcc = DCCEstimator(C_sample, R, D, alpha=0.01, beta=0.98)
dcc.fit()  # Fit DCC parameters
C_conditional = dcc.estimate()
```

### Unconditional Estimators

#### Sample Covariance

```python
from core.unconditional_estimator import SampleCov

estimator = SampleCov(R, D)
C = estimator.estimate()
```

#### Quadratic Inverse Shrinkage (QIS)

```python
from core.unconditional_estimator import QIS

estimator = QIS(R, D)
C = estimator.estimate()
```

#### Linear Shrinkage (LS)

```python
from core.unconditional_estimator import LS

# rho controls the shrinkage intensity (0 = no shrinkage, 1 = full shrinkage)
estimator = LS(R, D, rho=0.3)
C = estimator.estimate()
```

#### Average Oracle (AO)

```python
from core.unconditional_estimator import AO

estimator = AO(R, D, lookback_window=200)
estimator.fit(n_jobs=4)  # Optional: use multiprocessing
C = estimator.estimate()
```

### Conditional Estimators

#### Dynamic Conditional Correlation (DCC)

```python
from core.contional_estimator import DCCEstimator

# C: unconditional covariance matrix [n, n]
# R: return matrix [T, n]
# D: volatility matrix [T, n]
dcc = DCCEstimator(C, R, D, alpha=0.01, beta=0.98)

# Optionally fit parameters
dcc.fit()

# Estimate conditional covariance
C_conditional = dcc.estimate()
```

### Devolatization

#### GARCH-based Volatility Estimation

```python
from core.devolatizer import fit_garch_and_get_std

# Fit GARCH(1,1) models for each asset
D = fit_garch_and_get_std(
    R,
    mean="zero",      # Mean model: "zero" or "constant"
    vol="Garch",      # Volatility model: "Garch", "EGarch", "FIGarch"
    p=1,              # GARCH order p
    q=1,              # GARCH order q
    dist="normal"     # Error distribution: "normal" or "t"
)
```

## Project Structure

```
.
├── core/
│   ├── contional_estimator.py    # Conditional covariance estimators
│   ├── unconditional_estimator.py  # Unconditional covariance estimators
│   └── devolatizer.py            # GARCH-based devolatization
├── test/
│   ├── conditional_estimator_test.py
│   ├── unconditional_estimator_test.py
│   └── devolatizer_test.py
├── example/
│   └── example.ipynb             # Example notebooks
└── utils/
    └── preprocess.py             # Data preprocessing utilities
```

## Performance Optimization

The `AO` estimator supports multiprocessing for faster computation on large datasets:

```python
ao = AO(R, D, lookback_window=200)
ao.fit(n_jobs=4)  # Use 4 parallel processes
```

If `n_jobs=None` or `n_jobs<=1`, the estimator runs sequentially (default behavior).

## Testing

Run tests to verify the installation:

```bash
python test/unconditional_estimator_test.py
python test/conditional_estimator_test.py
python test/devolatizer_test.py
```

## References

- **DCC Model**: Engle, R. (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. *Journal of Business & Economic Statistics*, 20(3), 339-350.

- **QIS Method**: Ledoit, O., & Wolf, M. (2019). Quadratic shrinkage for large covariance matrices. *Bernoulli*, 25(4B), 3533-3565.

- **LS Method**: Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

- **AO Method**: Bongiorno, E. G., & Challet, D. (2024). Covariance matrix filtering and portfolio optimization. *arXiv preprint arXiv:2111.13109*.

## License

This project is part of the Baruch MFE Capstone program.

## Author

Baruch MFE Capstone Project

