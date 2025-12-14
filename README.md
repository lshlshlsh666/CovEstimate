# Comparison of DCC-based Covariance Matrix Estimation Methods

A comprehensive Python library for estimating covariance matrices in financial applications, implementing both unconditional and conditional covariance estimation methods. This project compares three state-of-the-art unconditional covariance estimators—Linear Shrinkage (LS), Quadratic Inverse Shrinkage (QIS), and Average Oracle (AO)—as priors for the Dynamic Conditional Correlation (DCC) model in large-scale portfolio optimization.

**Key Finding**: Our empirical evaluation on 500 U.S. equities (2000-2024) reveals that LS consistently delivers the strongest risk-adjusted performance in unconstrained settings, achieving an annualized Sharpe ratio of 1.58 with 81.5% cumulative return. This project is part of the Baruch MFE Capstone.

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

- **EWMA Volatility Estimation**: Exponentially weighted moving average (60-day window) for conditional volatility estimation
- **GARCH-based Devolatization**: Fit univariate GARCH models to estimate conditional volatilities for each asset (optional)

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

### Complete Workflow Example

The main workflow follows the empirical study in `main.ipynb`:

```python
import numpy as np
import pandas as pd
from core.unconditional_estimator import QIS, LS, AO
from core.contional_estimator import DCCEstimator

# 1. Load and prepare CRSP data
crsp = pd.read_csv("CRSP.csv")
crsp["RET"] = crsp["RET"].replace("C", np.nan).replace("B", np.nan).astype(float)
crsp = crsp.dropna(subset=["TICKER", "PRC"])

# 2. Create return matrix and select top 500 stocks by market cap
rets = pd.pivot_table(crsp, values="RET", index="date", columns="PERMNO")
companies = crsp[crsp.date == "2000-01-03"].nlargest(500, "MarketCap").PERMNO
rets = rets[companies]

# 3. Estimate volatility using EWMA (60-day window)
vol60 = rets.ewm(60).std() * np.sqrt(252) + 0.01

# 4. Split into train/test sets
train = rets.loc[:"2010-01-01"].dropna(axis=1)
test = rets.loc["2010-01-01":]
vol60 = vol60[train.columns].iloc[60:]
train = train.iloc[60:]

# 5. Estimate unconditional covariance matrices
C_qis = QIS(train.values, vol60.reindex_like(train).values).estimate()
C_ls = LS(train.values, vol60.reindex_like(train).values, rho=0.3).estimate()

# For AO, use extended dataset with sampling
S_total = train.values / vol60.reindex_like(train).values
C_ao = AO(train.values, vol60.reindex_like(train).values, 
          lookback_window=500).fit(S_total, sampling=10).estimate()

# 6. Run DCC-based GMV backtest
from main.ipynb import run_dcc_gmv_backtest

gmv_returns_ls, gmv_stats_ls, gmv_metrics_ls = run_dcc_gmv_backtest(
    C_ls, test_aligned, vol_test
)
```

### Basic Usage

```python
import numpy as np
from core.unconditional_estimator import SampleCov, QIS, LS, AO
from core.contional_estimator import DCCEstimator

# Generate sample return data
T, n = 1000, 10  # 1000 days, 10 assets
R = np.random.randn(T, n)  # Return matrix [T, n]

# Estimate volatility using EWMA (as in main study)
D = pd.DataFrame(R).ewm(60).std().values * np.sqrt(252) + 0.01

# Unconditional covariance estimation
qis = QIS(R, D)
C_qis = qis.estimate()

ls = LS(R, D, rho=0.3)
C_ls = ls.estimate()

ao = AO(R, D, lookback_window=200)
ao.fit(n_jobs=4)  # Use 4 parallel processes
C_ao = ao.estimate()

# Conditional covariance estimation (DCC)
dcc = DCCEstimator(C_ls, R, D, alpha=0.01, beta=0.98)
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

#### EWMA Volatility Estimation (Used in Main Study)

```python
import pandas as pd

# 60-day exponentially weighted moving average
vol60 = rets.ewm(60).std() * np.sqrt(252) + 0.01
# The 0.01 floor prevents division by near-zero values
```

#### GARCH-based Volatility Estimation (Optional)

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
│   ├── contional_estimator.py    # Conditional covariance estimators (DCC)
│   ├── unconditional_estimator.py  # Unconditional covariance estimators (LS, QIS, AO)
│   └── devolatizer.py            # GARCH-based devolatization
├── test/
│   ├── conditional_estimator_test.py
│   ├── unconditional_estimator_test.py
│   └── devolatizer_test.py
├── main.ipynb                    # Main empirical study (CRSP data, backtests)
├── example/
│   └── example.ipynb             # Additional example notebooks
├── utils/
│   └── preprocess.py             # Data preprocessing utilities
└── MastersProjectTemplate.tex   # LaTeX paper documenting the study
```

## Empirical Results

Our study evaluates three unconditional covariance estimators (LS, QIS, AO) as DCC priors using daily returns from the 500 largest U.S. equities (2000-2024) from the CRSP database.

### Unconstrained Portfolio Results

| Method | Sharpe Ratio | Cumulative Return | Max Drawdown | Turnover | Gross Leverage |
|--------|-------------|-------------------|--------------|----------|----------------|
| **LS** | **1.58** | **81.5%** | **-3.8%** | 0.27 | 3.39 |
| QIS | 1.41 | 66.0% | -3.8% | 0.31 | 3.81 |
| AO | 1.02 | 37.9% | -3.2% | 0.50 | 5.49 |

### Long-Only Portfolio Results

| Method | Sharpe Ratio | Cumulative Return | Max Drawdown | Turnover |
|--------|-------------|-------------------|--------------|----------|
| LS | 0.89 | 372.6% | -30.6% | 0.073 |
| QIS | 0.88 | 377.1% | -31.3% | 0.075 |
| AO | 0.82 | 391.4% | -33.0% | 0.086 |

**Key Findings:**
- LS delivers the strongest risk-adjusted performance in unconstrained settings
- QIS performs close to LS but requires slightly higher leverage and turnover
- AO exhibits lowest volatility but underperforms due to aggressive rebalancing
- In long-only portfolios, all three methods converge to similar performance

## Performance Optimization

The `AO` estimator supports multiprocessing and temporal sampling for faster computation on large datasets:

```python
ao = AO(R, D, lookback_window=500)
# Use sampling=10 to process every 10th time point
# Use n_jobs=4 for parallel processing
ao.fit(S_total, sampling=10, n_jobs=4)
```

- `sampling`: Process every Nth time point to reduce computation (default: 1, i.e., all points)
- `n_jobs`: Number of parallel processes (default: None, sequential processing)
- If `n_jobs=None` or `n_jobs<=1`, the estimator runs sequentially

## Data Requirements

The main empirical study (`main.ipynb`) requires CRSP equity data:
- Daily returns from CRSP database
- Fields: `RET`, `TICKER`, `PRC`, `SHROUT`, `CFACSHR`, `CFACPR`, `SHRCD`, `date`
- Data cleaning: Filters for common shares (SHRCD ∈ {10,11,12}), handles missing values
- Universe: Top 500 stocks by market capitalization on 2000-01-03

## Testing

Run tests to verify the installation:

```bash
python test/unconditional_estimator_test.py
python test/conditional_estimator_test.py
python test/devolatizer_test.py
```

## Implementation Notes

- **Volatility Estimation**: The main study uses 60-day EWMA, not GARCH (GARCH is available but optional)
- **Numerical Stability**: All covariance matrices are symmetrized; small eigenvalues are clipped at 1e-12
- **DCC Optimization**: Uses SLSQP algorithm with constraints α ≥ 0, β ≥ 0, α + β ≤ 1
- **Portfolio Optimization**: GMV portfolios use ridge regularization (1e-6) and Moore-Penrose pseudoinverse for numerical stability

## References

- **DCC Model**: Engle, R. (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. *Journal of Business & Economic Statistics*, 20(3), 339-350.

- **QIS Method**: Ledoit, O., & Wolf, M. (2019). Quadratic shrinkage for large covariance matrices. *Bernoulli*, 25(4B), 3533-3565.

- **LS Method**: Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.

- **AO Method**: Bongiorno, E. G., & Challet, D. (2024). Covariance matrix filtering and portfolio optimization. *arXiv preprint arXiv:2111.13109*.

## License

This project is part of the Baruch MFE Capstone program.

## Authors

- Yichen Li (yichen.li@baruch.cuny.edu)
- Shuhao Liu (shuhao.liu@baruch.cuny.edu)

Baruch MFE Capstone Project, 2024

