from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_ardl_short_long(effects_csv: Path, out_png: Path):
    """
    Plot short-run vs long-run effects from the ARDL model.
    """
    eff = pd.read_csv(effects_csv).set_index("Variable")[["Short_run", "Long_run"]]
    ax = eff.plot(kind="bar", figsize=(8, 4))
    ax.set_title("Short-run vs Long-run Effects (ARDL)")
    ax.set_ylabel("Effect size")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_dl_cumulative(cum_csv: Path, out_png: Path):
    """
    Plot cumulative (long-run) effects from the DL model.
    """
    df = pd.read_csv(cum_csv).set_index("Variable")
    ax = df.plot(kind="bar", figsize=(7, 4))
    ax.set_title("Cumulative (Long-run) Effects from DL(L=2)")
    ax.set_ylabel("Sum of coefficients")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_fe_coeffs(coeffs_csv: Path, out_png: Path):
    """
    Plot fixed-effects model coefficients for visual inspection.
    """
    df = pd.read_csv(coeffs_csv).set_index("Variable")
    ax = df["Coefficient"].plot(kind="bar", figsize=(7, 4))
    ax.set_title("Fixed-effects Coefficients (within-transformed OLS)")
    ax.set_ylabel("Coefficient")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
