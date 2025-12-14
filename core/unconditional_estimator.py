from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


def _process_ao_index(cov_now: np.ndarray, cov_next: np.ndarray) -> np.ndarray:
    """
    Process a single index for AO model fitting.

    Parameters
    ----------
    cov_now : np.ndarray
        Current covariance matrix
    cov_next : np.ndarray
        Next covariance matrix

    Returns
    -------
    d : np.ndarray
        Computed d vector
    """
    _, eigenvectors = np.linalg.eigh(cov_now)
    d = []
    for i in range(eigenvectors.shape[1]):
        eigenvector = eigenvectors[:, i]
        d.append(eigenvector @ (cov_next @ eigenvector))
    return np.array(d)


class UnconditionalEstimator(ABC):
    """
    Abstract base class for all estimators to estimate the unconditional covariance matrix.
    """

    def __init__(self, R: np.ndarray, D: np.ndarray):
        self.R = R  # return matrix -- [T, n], ascending by time
        self.D = D  # volatility matrix -- [T, n], ascending by time
        self.S = R / D  # devolatilized return matrix -- [T, n], ascending by time

    @abstractmethod
    def estimate(self) -> np.ndarray:
        """
        Estimate the unconditional covariance matrix.
        """


class SampleCov(UnconditionalEstimator):
    """
    Directly use the sample covariance matrix.
    """

    def estimate(self) -> np.ndarray:
        return np.cov(self.S, rowvar=False)


class QIS(UnconditionalEstimator):
    """
    Estimate the unconditional covariance matrix using the QIS model.
    Reference: http://www.ledoit.net/BEJ1911-021R1A0.pdf
    """

    def estimate(self) -> np.ndarray:
        n, p = self.S.shape
        q = p / n
        sample_cov = np.cov(self.S, rowvar=False, bias=True)  # [n, n]
        eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)  # [n], [n, n]

        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        lam_inv = 1.0 / eigenvalues

        h = n ** (-1.0 / 3.0)

        def theta_hat(x):
            num = lam_inv * (lam_inv - x)
            den = (lam_inv - x) ** 2 + (h**2) * lam_inv**2
            return (1.0 / p) * np.sum(num / den)

        def A2_hat(x):
            term1 = lam_inv * (lam_inv - x) / ((lam_inv - x) ** 2 + (h**2) * lam_inv**2)
            term2 = lam_inv * (h * lam_inv) / ((lam_inv - x) ** 2 + (h**2) * lam_inv**2)
            return ((1.0 / p) * np.sum(term1)) ** 2 + ((1.0 / p) * np.sum(term2)) ** 2

        delta_inv = (
            (1.0 - q) ** 2 * lam_inv
            + 2.0 * q * (1.0 - q) * lam_inv * np.array([theta_hat(x) for x in lam_inv])
            + q**2 * lam_inv * np.array([A2_hat(x) for x in lam_inv])
        )

        delta = 1.0 / delta_inv
        delta = np.clip(delta, 1e-12, None)

        cov_QIS = eigenvectors @ np.diag(delta) @ eigenvectors.T
        cov_QIS = 0.5 * (cov_QIS + cov_QIS.T)

        return cov_QIS


class LS(UnconditionalEstimator):
    """
    Estimate the unconditional covariance matrix using the linear shrinkage method.
    Reference: http://www.ledoit.net/honey.pdf
    """

    def __init__(self, R: np.ndarray, D: np.ndarray, rho: float):
        super().__init__(R, D)
        self.rho = rho

    def estimate(self) -> np.ndarray:
        sample_cov = np.cov(self.S, rowvar=False, bias=True)  # [n, n]
        eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)  # [n], [n, n]
        mean_eigenvalue = np.mean(eigenvalues)
        shrinkaged_eigenvalues = self.rho * mean_eigenvalue + (1 - self.rho) * eigenvalues
        return eigenvectors @ np.diag(shrinkaged_eigenvalues) @ eigenvectors.T


class AO(UnconditionalEstimator):
    """
    Estimate the unconditional covariance matrix using the AO method.
    Reference: https://arxiv.org/pdf/2111.13109
    """

    def __init__(self, R: np.ndarray, D: np.ndarray, lookback_window: int):
        super().__init__(R, D)
        self.lookback_window = lookback_window
        self.d: np.ndarray = None

    def fit(self, S: np.ndarray | None = None, n_jobs: int | None = None, sampling: int = 10) -> "AO":
        """
        Fit the AO model -- Need to use a longer time period to estimate the average oracle eigenvalues.
        However, if S is not provided, just use self.S to fit the model.

        Parameters
        ----------
        S : np.ndarray | None
            Devolatilized return matrix. If None, use self.S.
        n_jobs : int | None
            Number of parallel jobs. If None, use all available CPUs.
        sampling : int
            Sampling interval to reduce computation. Default is 10.
        """
        if S is None:
            S = self.S
        rolling_cov = pd.DataFrame(S).rolling(window=self.lookback_window).cov().loc[self.lookback_window :]
        indices = rolling_cov.index.get_level_values(0).unique()[:-1:sampling]

        # Convert rolling_cov to numpy arrays for multiprocessing
        # Store as list of tuples (cov_now, cov_next) for easier parallel processing
        cov_pairs = []
        for index in indices:
            cov_now = rolling_cov.loc[index].values
            cov_next = rolling_cov.loc[index + 1].values
            cov_pairs.append((cov_now, cov_next))

        d_sum = np.zeros(S.shape[1])  # [n]
        cnt = 0

        # Use multiprocessing if n_jobs is specified and > 1
        if n_jobs is not None and n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                futures = [executor.submit(_process_ao_index, cov_now, cov_next) for cov_now, cov_next in cov_pairs]

                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Fitting AO model..."):
                    d = future.result()
                    d_sum += d
                    cnt += 1
        else:
            # Sequential processing (original behavior)
            for cov_now, cov_next in tqdm(cov_pairs, desc="Fitting AO model..."):
                d = _process_ao_index(cov_now, cov_next)
                d_sum += d
                cnt += 1

        self.d = d_sum / cnt
        return self

    def estimate(self) -> np.ndarray:
        """
        Estimate the unconditional covariance matrix using the AO model.
        """
        if self.d is None:
            raise ValueError("AO model not fitted yet. Please fit the model first.")
        sample_cov = np.cov(self.S, rowvar=False, bias=True)  # [n, n]
        _, eigenvectors = np.linalg.eigh(sample_cov)  # [n], [n, n]
        return eigenvectors @ np.diag(self.d) @ eigenvectors.T
