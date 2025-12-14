from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize


class ConditionalEstimator(ABC):
    """
    Abstract base class for all estimators to estimate the conditional covariance matrix.
    """

    def __init__(self, C: np.ndarray, R: np.ndarray, D: np.ndarray):
        self.C = C  # unconditional covariance matrix -- [n, n]
        self.R = R  # return matrix -- [T, n], ascending by time
        self.D = D  # volatility matrix -- [T, n], ascending by time
        self.S = R / D  # devolatilized return matrix -- [T, n], ascending by time

    @abstractmethod
    def estimate(self) -> np.ndarray:
        """
        Estimate the covariance matrix from unconditional covariance matrix.
        """


class RawEstimator(ConditionalEstimator):
    """
    Do nothing! Just return the input unconditional covariance matrix.
    """

    def estimate(self) -> np.ndarray:
        return self.C


class DCCEstimator(ConditionalEstimator):
    """
    Estimate the covariance matrix using the DCC model.
    Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1296428
    """

    def __init__(self, C: np.ndarray, R: np.ndarray, D: np.ndarray, alpha: float = 0.01, beta: float = 0.98):
        super().__init__(C, R, D)
        self.alpha = alpha
        self.beta = beta

        # dynamic variables
        self.ts: int = 0
        self.Q_t: np.ndarray = C
        self.R_t: np.ndarray = C

    def reset(self):
        self.ts = 0
        self.Q_t = self.C
        self.R_t = self.C

    def _step(self, alpha: float | None = None, beta: float | None = None) -> np.ndarray:
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        s_t = self.S[self.ts]
        self.Q_t = (1 - alpha - beta) * self.C + alpha * np.outer(s_t, s_t) + beta * self.Q_t
        factor = np.sqrt(np.diag(1 / np.diag(self.Q_t)))
        self.R_t = factor @ self.Q_t @ factor
        self.ts += 1
        return s_t

    def _objective(self, args: list[float]) -> float:
        alpha, beta = args
        self.reset()
        loss = 0
        while self.ts < len(self.S):
            s_t = self._step(alpha, beta)
            loss += np.log(np.linalg.det(self.R_t)) + s_t @ (self.R_t @ s_t)
        return loss

    def fit(self):
        constraints = [
            {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
        ]
        result = minimize(
            self._objective, [self.alpha, self.beta], method='SLSQP', constraints=constraints, bounds=[(0, 1), (0, 1)]
        )
        self.alpha = result.x[0]
        self.beta = result.x[1]

    def estimate(self) -> np.ndarray:
        self.reset()
        while self.ts < len(self.S):
            self._step()

        D_t = np.diag(self.D[-1])

        return D_t @ self.R_t @ D_t
