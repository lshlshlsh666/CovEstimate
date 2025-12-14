import numpy as np
from arch import arch_model


def fit_garch_and_get_std(
    R: np.ndarray,
    mean: str = "zero",
    vol: str = "Garch",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> np.ndarray:
    """
    Fit univariate GARCH models for each column in returns
    and return conditional standard deviations.

    Parameters
    ----------
    R : np.ndarray
        Asset returns [T, n]
    mean : str
        Mean model ("zero", "constant")
    vol : str
        Volatility model ("Garch", "EGarch", "FIGarch")
    p, q : int
        GARCH orders
    dist : str
        Error distribution ("normal", "t")
    """
    T, n = R.shape
    std = np.zeros((T, n))

    # Fit GARCH model for each asset (column)
    for i in range(n):
        am = arch_model(
            R[:, i],  # Extract single time series for asset i
            mean=mean,
            vol=vol,
            p=p,
            q=q,
            dist=dist,
            rescale=True,
        )
        res = am.fit(disp="off")
        std[:, i] = res.conditional_volatility / res.scale

    return std
