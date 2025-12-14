import numpy as np
import pandas as pd

from core.unconditional_estimator import AO, LS, QIS, SampleCov


def test_unconditional_estimator():
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
    print("True Covariance Matrix:")
    print(f"  - Condition Number: {np.linalg.cond(C_true)}")

    # Volatility matrix (use rolling window to estimate)
    D = pd.DataFrame(log_returns).rolling(window=warmup_period).std().to_numpy()[warmup_period:]
    D_t = np.diag(D[-1])

    # Unconditional covariance matrix
    C = np.cov(R / D, rowvar=False)
    C_rescale = D_t @ C @ D_t

    print("Unconditional Covariance Matrix:")
    print(
        "  - Frobenius norm relative error of the difference with C_true: "
        + f"{np.linalg.norm(C_rescale - C_true) / np.linalg.norm(C_true)}"
    )
    print(f"  - Conditional Number relative : {np.linalg.cond(C_rescale)}")

    for estimator in [
        SampleCov(R, D),
        QIS(R, D),
        LS(R, D, 0.0),
        LS(R, D, 0.1),
        LS(R, D, 0.3),
        LS(R, D, 0.5),
        AO(R, D, 100),
        AO(R, D, 200),
    ]:
        print(f"Estimator: {estimator.__class__.__name__}")
        if estimator.__class__ == AO:
            print(f"  - Lookback window: {estimator.lookback_window}")
            estimator.fit()
        elif estimator.__class__ == LS:
            print(f"  - Rho: {estimator.rho}")

        cov = estimator.estimate()
        cov_rescale = D_t @ cov @ D_t

        print(
            "  - Frobenius norm relative error of the difference with C_true: "
            + f"{np.linalg.norm(cov_rescale - C_true) / np.linalg.norm(C_true)}"
        )
        print(
            "  - Frobenius norm relative error of the difference with C_sample: "
            + f"{np.linalg.norm(cov_rescale - C_rescale) / np.linalg.norm(C_rescale)}"
        )
        print(f"  - Conditional Number: {np.linalg.cond(cov_rescale)}")


if __name__ == "__main__":
    test_unconditional_estimator()
