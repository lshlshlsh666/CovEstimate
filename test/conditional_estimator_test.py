import numpy as np
import pandas as pd

from core.contional_estimator import DCCEstimator


def test_dcc_estimator():
    np.random.seed(666)
    T = 1000  # number of days
    warmup_period = 100
    n = 10  # number of assets

    sigma = 0.2
    dt = 1 / 252

    # Generate price paths
    raw_log_returns = np.random.normal(loc=0, scale=1, size=(T, n))
    C_root_true = np.random.randn(n, n) * sigma * np.sqrt(dt)
    C_true = C_root_true.T @ C_root_true
    log_returns = raw_log_returns @ C_root_true.T
    R = log_returns[warmup_period:]

    # Volatility matrix (use rolling window to estimate)
    D = pd.DataFrame(log_returns).rolling(window=warmup_period).std().to_numpy()[warmup_period:]

    # Unconditional covariance matrix
    C = np.cov(log_returns[warmup_period:], rowvar=False)

    # Instantiate and estimate using DCCEstimator
    model = DCCEstimator(C, R, D, alpha=0.05, beta=0.9)
    print(f"Before fitting: alpha = {model.alpha}, beta = {model.beta}")
    model.fit()
    print(f"After fitting: alpha = {model.alpha}, beta = {model.beta}")
    conditional_cov = model.estimate()

    print(
        "Frobenius norm relative error of the difference with C_true: "
        + f"{np.linalg.norm(conditional_cov - C_true) / np.linalg.norm(C_true)}"
    )
    print(
        "Frobenius norm relative error of the difference with C_sample: "
        + f"{np.linalg.norm(conditional_cov - C) / np.linalg.norm(C)}"
    )


if __name__ == "__main__":
    test_dcc_estimator()
