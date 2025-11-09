"""
Run OLS for Sydney & Melbourne with diagnostics and charts.
Outputs -> outputs/
  - report_M1_city.txt / report_M2_city.txt
  - coef_M1_city.csv   / coef_M2_city.csv
  - figure_prices_cash.png
  - figure_fit_M1_city.png / figure_resid_M1_city.png (M2 同理)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
from housing._diag_utils import make_full_rank
from housing.config import CLEANED, OUTPUTS

OUTPUTS.mkdir(parents=True, exist_ok=True)

PANEL = CLEANED / "housing_model_quarterly_panel.csv"


def drop_bad_cols(df):
    # 丢掉全空或只有一个唯一值的列
    good = [c for c in df.columns if df[c].notna().sum()>5 and df[c].nunique(dropna=True)>1]
    return df[good]


def pct_change_log(x, periods=1):
    x = pd.to_numeric(x, errors="coerce")
    return np.log(x).diff(periods)

def standardize_cols(df):
    # 保证这些列存在即便为 NaN
    for c in ["approvals_units_nsw","approvals_units_vic",
              "population_nsw","population_vic",
              "emp_total_nsw","emp_total_vic",
              "cash_rate","hpi_sydney","hpi_melbourne"]:
        if c not in df.columns: df[c] = np.nan
    return df

def prepare_panel():
    df = pd.read_csv(PANEL)
    df = standardize_cols(df)
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q").astype(str)
    # 排序并建立 city-wise 宽表 → 拼成长表（city 维度）
    keep = ["quarter","cash_rate",
            "approvals_units_nsw","approvals_units_vic",
            "population_nsw","population_vic",
            "emp_total_nsw","emp_total_vic",
            "hpi_sydney","hpi_melbourne"]
    df = df[keep].drop_duplicates("quarter").sort_values("quarter").reset_index(drop=True)

    long = pd.DataFrame({
        "quarter": np.concatenate([df["quarter"].values, df["quarter"].values]),
        "city":    ["Sydney"]*len(df) + ["Melbourne"]*len(df),
        "hpi":     np.concatenate([df["hpi_sydney"].values, df["hpi_melbourne"].values]),
        "approvals": np.concatenate([df["approvals_units_nsw"].values, df["approvals_units_vic"].values]),
        "population": np.concatenate([df["population_nsw"].values, df["population_vic"].values]),
        "employment": np.concatenate([df["emp_total_nsw"].values, df["emp_total_vic"].values]),
        "cash_rate": np.concatenate([df["cash_rate"].values, df["cash_rate"].values]),
    })
    long = long.replace([np.inf,-np.inf], np.nan)
    return long

def run_ols(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop").fit()
    # Newey-West (HAC, lag=4 ≈ 一年)
    hac = model.get_robustcov_results(cov_type="HAC", maxlags=4)
    return model, hac


def diag_tests(model):
    import numpy as np
    y = model.model.endog
    X = model.model.exog

    # White / BP 需要满秩X：做简易清洗 + QR降维
    Xr = make_full_rank(np.asarray(X, dtype=float))
    # 残差与X需对齐长度：取有效行
    resid = model.resid
    m = min(len(resid), len(Xr))
    if m == 0:
        return {"BP":{"LM stat":np.nan,"p":np.nan},
                "White":{"LM stat":np.nan,"p":np.nan},
                "BG(4)":{"LM stat":np.nan,"p":np.nan}}

    # 统计量
    bp = het_breuschpagan(resid[-m:], Xr[-m:, :])
    try:
        wt = het_white(resid[-m:], Xr[-m:, :])
        white = {"LM stat": wt[0], "p": wt[1]}
    except AssertionError:
        white = {"LM stat": np.nan, "p": np.nan}
    bg = acorr_breusch_godfrey(model, nlags=4)

    return {"BP":{"LM stat": bp[0], "p": bp[1]},
            "White": white,
            "BG(4)":{"LM stat": bg[0], "p": bg[1]}}
def save_report(name, model, hac, tests):
    txt = []
    txt.append(f"=== {name} OLS (classical SE) ===\n{model.summary()}\n")
    txt.append(f"=== {name} OLS (HAC Newey-West, lag=4) ===\n{hac.summary()}\n")
    txt.append("=== Residual diagnostics ===")
    for k,v in tests.items():
        txt.append(f"{k}: {v}")
    (OUTPUTS / f"report_{name}.txt").write_text("\n".join(map(str, txt)), encoding="utf-8")

    # 系数表（便于拉进报告）
    coefs = pd.DataFrame({
        "param": hac.params.index,
        "coef": hac.params.values,
        "std_err_hac": hac.bse.values,
        "t_hac": hac.tvalues.values,
        "p_hac": hac.pvalues.values,
    })
    coefs.to_csv(OUTPUTS / f"coef_{name}.csv", index=False)

def plot_series(panel):
    import matplotlib.pyplot as plt
    # 价格指数（两城）+ 现金利率（右轴, 线性缩放以同图展示）
    df_s = panel.pivot(index="quarter", columns="city", values="hpi")
    cash = panel.drop_duplicates("quarter").set_index("quarter")["cash_rate"]
    scale = df_s.max().max() / (cash.max() if pd.notna(cash.max()) else 1)
    plt.figure()
    df_s.plot(ax=plt.gca())
    (cash*scale).plot(ax=plt.gca(), linestyle="--")
    plt.title("House Price Index (Syd/Mel) & Cash Rate (scaled)")
    plt.xlabel("Quarter"); plt.ylabel("Index (2011Q4=100, approx)")
    plt.legend(list(df_s.columns)+["cash_rate (scaled)"])
    plt.tight_layout()
    plt.savefig(OUTPUTS / "figure_prices_cash.png", dpi=160)
    plt.close()

def plot_fit(name, model, panel_city):
    import matplotlib.pyplot as plt
    fit = pd.Series(model.fittedvalues, index=panel_city.dropna().index)
    y = panel_city.loc[fit.index, "y"]
    q = panel_city.loc[fit.index, "quarter"]

    # 真实 vs 拟合
    plt.figure()
    plt.plot(q, y, label="actual")
    plt.plot(q, fit, label="fitted", linestyle="--")
    plt.title(f"{name}: Actual vs Fitted")
    plt.xlabel("Quarter"); plt.ylabel("Dependent")
    plt.xticks(rotation=60); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUTS / f"figure_fit_{name}.png", dpi=160); plt.close()

    # 残差
    plt.figure()
    plt.plot(q, y - fit, label="residuals")
    plt.axhline(0, lw=1)
    plt.title(f"{name}: Residuals over time")
    plt.xlabel("Quarter"); plt.ylabel("Residual")
    plt.xticks(rotation=60); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUTS / f"figure_resid_{name}.png", dpi=160); plt.close()

def main():
    panel = prepare_panel()
    plot_series(panel)  # 总览图

    results_summary = []
    for city in ["Sydney","Melbourne"]:
        sub = panel[panel["city"]==city].copy()
        sub["quarter_num"] = range(len(sub))  # 可选趋势控制

        # --- 模型 M1（水平） ---
        sub["y"] = pd.to_numeric(sub["hpi"], errors="coerce")
        X1 = drop_bad_cols(sub[["cash_rate","approvals","population","employment","quarter_num"]])
        m1, hac1 = run_ols(sub["y"], X1)
        t1 = diag_tests(m1)
        name1 = f"M1_{city}"
        save_report(name1, m1, hac1, t1)
        plot_fit(name1, m1, sub)

        # --- 模型 M2（增长，log-diff） ---
        sub["y"] = pct_change_log(sub["hpi"])
        X2 = drop_bad_cols(pd.DataFrame({
            "d_cash": sub["cash_rate"].diff(),
            "dln_approvals": pct_change_log(sub["approvals"]),
            "dln_population": pct_change_log(sub["population"]),
        })
        m2, hac2 = run_ols(sub["y"].iloc[1:], X2.iloc[1:])  # 丢第一行 NaN
        t2 = diag_tests(m2)
        name2 = f"M2_{city}"
        save_report(name2, m2, hac2, t2)
        plot_fit(name2, m2, pd.concat([sub[["quarter","y"]], X2], axis=1).iloc[1:])

    print("✅ Done. See outputs/ for reports, coefs, and figures.")

if __name__ == "__main__":
    main()
