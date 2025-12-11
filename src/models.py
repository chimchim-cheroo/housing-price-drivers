from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------- helper functions ----------
def within_transform(df, y, X_cols, group="region"):
    d = df[[group, "quarter", y] + X_cols].dropna().copy()
    for col in [y] + X_cols:
        d[col + "_dm"] = d[col] - d.groupby(group)[col].transform("mean")
    y_dm = d[y + "_dm"]
    X_dm = sm.add_constant(d[[c + "_dm" for c in X_cols]])
    return d, y_dm, X_dm


def add_lags(df, cols, lags=2, group="region"):
    d = df.sort_values([group, "quarter"]).copy()
    for c in cols:
        for k in range(1, lags + 1):
            d[f"{c}_L{k}"] = d.groupby(group)[c].shift(k)
    return d


def add_diff_and_lag(df, cols, group="region"):
    d = df.sort_values([group, "quarter"]).copy()
    for c in cols:
        d[f"d_{c}"] = d.groupby(group)[c].diff()
        d[f"{c}_L1"] = d.groupby(group)[c].shift(1)
    return d


def fit_ols_hac(y, X, maxlags=1):
    yv = np.asarray(y, dtype=float)
    Xv = np.asarray(X, dtype=float)
    if Xv.ndim == 1:
        Xv = Xv.reshape(-1, 1)
    mask = ~np.isnan(yv) & ~np.isnan(Xv).any(axis=1)
    yv, Xv = yv[mask], Xv[mask]
    res = sm.OLS(yv, Xv, missing="drop").fit()
    return res.get_robustcov_results(cov_type="HAC", maxlags=maxlags)


# ---------- main model functions ----------
def run_fe(df_long: pd.DataFrame, out_dir: Path):
    df = df_long.copy()

    rename_map = {
        "cash_rate_qavg": "cash_rate_level",
        "cash_rate": "cash_rate_level",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    candidates = [
        "cash_rate_level",
        "approvals_units",
        "wealth",
        "avg_net_wealth_k",
        "income",
        "unemp_rate",
        "part_rate",
        "ln_population",
        "rate_x_supply_gap",
    ]
    base_X = [c for c in candidates if c in df.columns]

    if not base_X:
        raise ValueError(
            "FE model found no regressors. Available columns:\n"
            f"{list(df.columns)}"
        )

    d, y_dm, X_dm = within_transform(df, "ln_hpi", base_X)
    fe = fit_ols_hac(y_dm, X_dm, maxlags=1)
    (out_dir / "fe_ols_summary.txt").write_text(fe.summary().as_text())

    coef_df = pd.DataFrame({"Variable": X_dm.columns, "Coefficient": fe.params})
    coef_df.to_csv(out_dir / "fe_coeffs.csv", index=False)

    return fe, base_X


def run_dl(df_long: pd.DataFrame, out_dir: Path, vars_):
    dlag = add_lags(df_long, vars_, lags=2)
    X_cols = sum(([v, f"{v}_L1", f"{v}_L2"] for v in vars_), [])
    d, y, X = within_transform(dlag, "ln_hpi", X_cols)
    dl = fit_ols_hac(y, X, maxlags=1)
    (out_dir / "DL_model_summary.txt").write_text(dl.summary().as_text())

    params = dict(zip(dl.model.exog_names, dl.params))
    rows = []
    for v in vars_:
        cumulative = sum(params.get(f"{v}_dm{k}", 0.0) for k in ["", "_L1", "_L2"])
        rows.append([v, cumulative])

    pd.DataFrame(rows, columns=["Variable", "DL_cumulative"]).to_csv(
        out_dir / "DL_longrun_effects.csv", index=False
    )

    return dl


def run_ardl(df_long: pd.DataFrame, out_dir: Path, vars_):
    d = add_diff_and_lag(df_long, ["ln_hpi"] + vars_)
    X_cols = [f"d_{v}" for v in vars_] + ["ln_hpi_L1"] + [f"{v}_L1" for v in vars_]
    _, y, X = within_transform(d, "d_ln_hpi", X_cols)
    ar = fit_ols_hac(y, X, maxlags=1)
    (out_dir / "ARDL_model_summary.txt").write_text(ar.summary().as_text())

    params = dict(zip(ar.model.exog_names, ar.params))
    rows = []
    for v in vars_:
        short_run = params.get(f"d_{v}_dm", np.nan)
        long_run = -params.get(f"{v}_L1_dm", np.nan) / params.get("ln_hpi_L1_dm", np.nan)
        rows.append([v, short_run, long_run])

    pd.DataFrame(rows, columns=["Variable", "Short_run", "Long_run"]).to_csv(
        out_dir / "ARDL_effects_summary.csv", index=False
    )

    return ar
