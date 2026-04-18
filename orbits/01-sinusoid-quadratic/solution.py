"""Closed-form solution for the toy symbolic-regression benchmark.

Coefficients eyeballed from train_data.csv (no fitting).

Derivation:
  Hypothesis: y = A*sin(B*x + C) + D*x**2 + E.

  Endpoint anchor (B=1, C=0 trial):
    y(-4) - y(+4) = 2*A*sin(-4)                (quadratic and constant cancel)
    observed:    2.372 - 0.849 = 1.523
    => A = 1.523 / (2 * 0.7568) = 1.006        --> A = 1.

  Origin anchor:
    y(0) approx 0 => E = 0.

  Quadratic envelope:
    y(+4) - sin(4) = 16*D      =>  D = (0.849 - (-0.757)) / 16 = 0.1004
    y(-4) - sin(-4) = 16*D     =>  D = (2.372 - 0.757) / 16 = 0.1009
    Both agree at D = 0.1.

Final closed form:
    f(x) = sin(x) + 0.1 * x**2
"""

import numpy as np


def f(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(x) + 0.1 * x**2
