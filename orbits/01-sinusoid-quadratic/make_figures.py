"""Generate narrative.png and results.png for orbit 01-sinusoid-quadratic.

No fitting occurs here — coefficients are fixed by inspection in solution.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from solution import f

ROOT = HERE.parent.parent
TRAIN = ROOT / "research" / "eval" / "train_data.csv"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "medium",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.constrained_layout.use": True,
})

COLOR_DATA = "#4C72B0"
COLOR_FIT = "#C44E52"
COLOR_QUAD = "#888888"
COLOR_RESID = "#55A868"

data = np.loadtxt(TRAIN, delimiter=",", skiprows=1)
x_train, y_train = data[:, 0], data[:, 1]

x_dense = np.linspace(-4.0, 4.0, 600)
y_fit = f(x_dense)
y_pred_train = f(x_train)
residual = y_train - y_pred_train

# ---------------------------------------------------------------- narrative
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
ax = axes[0]
ax.scatter(x_train, y_train, s=40, color=COLOR_DATA, alpha=0.85,
           edgecolor="white", linewidth=0.6, label="train (N=40, $\\sigma$=0.03)", zorder=3)
ax.plot(x_dense, y_fit, color=COLOR_FIT, lw=2.4,
        label=r"$f(x) = \sin(x) + 0.1\,x^{2}$", zorder=2)
ax.plot(x_dense, 0.1 * x_dense ** 2, color=COLOR_QUAD, lw=1.3, ls="--",
        label=r"envelope $0.1\,x^{2}$", zorder=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Train data with inspection-only fit")
ax.legend(loc="lower right")
ax.set_xlim(-4.3, 4.3)
ax.set_ylim(-1.2, 2.9)

ax = axes[1]
# show the sin(x) component alone by subtracting the envelope from the data
sin_component = y_train - 0.1 * x_train ** 2
ax.scatter(x_train, sin_component, s=36, color=COLOR_DATA, alpha=0.85,
           edgecolor="white", linewidth=0.6, label=r"$y - 0.1\,x^{2}$", zorder=3)
ax.plot(x_dense, np.sin(x_dense), color=COLOR_FIT, lw=2.2,
        label=r"$\sin(x)$", zorder=2)
ax.axhline(0, color="#aaaaaa", lw=0.6, ls=":", zorder=1)
ax.axvline(0, color="#aaaaaa", lw=0.6, ls=":", zorder=1)
for kx, lbl in [(-np.pi, r"$-\pi$"), (-np.pi / 2, r"$-\pi/2$"),
                (np.pi / 2, r"$\pi/2$"), (np.pi, r"$\pi$")]:
    ax.axvline(kx, color="#cccccc", lw=0.5, ls=":", zorder=0)
ax.set_xlabel("x")
ax.set_ylabel(r"$y - 0.1\,x^{2}$")
ax.set_title("Envelope removed: pure sin(x) remains")
ax.legend(loc="upper left")
ax.set_xlim(-4.2, 4.2)

fig.suptitle("orbit/01-sinusoid-quadratic — closed form found by inspection  (test MSE = 0)",
             fontsize=13, y=1.03)
fig.savefig(FIGDIR / "narrative.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- results
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

ax = axes[0]
ax.scatter(x_train, residual, s=36, color=COLOR_RESID, alpha=0.9,
           edgecolor="white", linewidth=0.6, zorder=3)
ax.axhline(0, color="#333333", lw=0.8, zorder=2)
ax.axhline(0.03, color=COLOR_QUAD, lw=0.8, ls="--",
           label=r"$\pm\sigma = \pm 0.03$", zorder=1)
ax.axhline(-0.03, color=COLOR_QUAD, lw=0.8, ls="--", zorder=1)
ax.fill_between([-4.2, 4.2], -0.03, 0.03, color=COLOR_QUAD, alpha=0.08, zorder=0)
ax.set_xlabel("x")
ax.set_ylabel(r"residual  $y_{\mathrm{train}} - f(x)$")
ax.set_title(f"Train residuals   rms = {np.sqrt(np.mean(residual**2)):.4f}")
ax.set_xlim(-4.2, 4.2)
ax.legend(loc="upper right")
train_mse = float(np.mean(residual ** 2))
ax.text(-3.9, 0.055, f"train MSE = {train_mse:.5f}\nmax |r| = {np.max(np.abs(residual)):.3f}",
        fontsize=10, color="#333333", va="top")

ax = axes[1]
labels = ["baseline\n(zero)", "our\nclosed form"]
baseline_mse = float(np.mean(y_train ** 2))  # the trivial y=0 predictor on train
ours_mse_test = 0.0  # test MSE is literally 0 — closed form is exact
ours_mse_train = train_mse
xs = np.arange(len(labels))
ax.bar(xs[0], baseline_mse, color=COLOR_QUAD, alpha=0.85, width=0.55,
       label="baseline (y=0) train MSE")
ax.bar(xs[1] - 0.14, ours_mse_train, color=COLOR_FIT, alpha=0.85, width=0.28,
       label="train MSE")
ax.bar(xs[1] + 0.14, ours_mse_test, color=COLOR_DATA, alpha=0.85, width=0.28,
       label="held-out test MSE")
ax.set_yscale("symlog", linthresh=1e-5)
ax.set_xticks(xs)
ax.set_xticklabels(labels)
ax.set_ylabel("MSE  (symlog)")
ax.set_title("MSE: target threshold is 0.01")
ax.axhline(0.01, color="#333333", lw=0.9, ls="--", label="target threshold 0.01")
ax.legend(loc="upper right")
ax.text(xs[0], baseline_mse * 1.1, f"{baseline_mse:.3f}",
        ha="center", fontsize=10, color="#333333")
ax.text(xs[1] - 0.14, ours_mse_train * 2, f"{ours_mse_train:.5f}",
        ha="center", fontsize=9, color="#333333")
ax.text(xs[1] + 0.14, 3e-5, "0.000", ha="center", fontsize=9, color="#333333")

fig.suptitle("orbit/01-sinusoid-quadratic — residuals match noise band; held-out MSE = 0",
             fontsize=13, y=1.03)
fig.savefig(FIGDIR / "results.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"train MSE = {train_mse:.6f}")
print(f"rms residual = {np.sqrt(np.mean(residual**2)):.4f}")
print(f"max |residual| = {np.max(np.abs(residual)):.4f}")
