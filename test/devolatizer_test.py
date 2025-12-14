import numpy as np

from core.devolatizer import fit_garch_and_get_std


def test_devolatizer():
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
    D_true = np.diag(np.sqrt(np.diag(C_true)))
    log_returns = raw_log_returns @ C_root_true.T
    R = log_returns[warmup_period:]
    # Volatility matrix (use rolling window to estimate)
    D = fit_garch_and_get_std(R)
    D_t = np.diag(D[-1])

    print(
        "Frobenius norm relative error of the difference with D_true: "
        + f"{np.linalg.norm(D_t - D_true) / np.linalg.norm(D_true)}"
    )


if __name__ == "__main__":
    test_devolatizer()
