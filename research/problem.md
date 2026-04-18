# Toy Symbolic Regression

## Problem Statement

Given `research/eval/train_data.csv` — 40 noisy (x, y) points on x ∈ [-4, 4] with noise σ=0.03 — propose a closed-form f(x) that best fits the underlying clean function.

Constraints:
- Write the solution as Python code in `solution.py` exporting `f(x: np.ndarray) -> np.ndarray`.
- NO sklearn, NO fitting loops, NO scipy.optimize — just the symbolic expression.
- Tune coefficients by inspection only (eyeball, not curve_fit).
- Evaluator at `research/eval/evaluator.py` — do NOT rebuild it.

## Solution Interface

`solution.py` must define `f(x: np.ndarray) -> np.ndarray`. The evaluator calls `f(x_test)` on a held-out clean test set covering x ∈ [-4, 4].

## Success Metric

MSE on held-out test set (minimize). Target: MSE < 0.01. Budget: max 2 orbits.
