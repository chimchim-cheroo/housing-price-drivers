"""
Build quarterly panel for Sydney & Melbourne (2008Q1–latest)
Inputs (from data/cleaned/ unless specified):
  - house_price_index_SydMel_2008Q1_2024Q2.csv  (long: quarter, city, index)
  - population_NSW_VIC_quarterly_2008_2025_wide.csv
  - approvals_quarterly_states_long_2008_onwards.csv
  - nsw_labour_quarterly_2008_onwards.csv
  - vic_labour_quarterly_2008_onwards.csv
  - cash_rate_quarterly.csv
  - wealth and income.csv   (in data/cleaned/ or data/raw/; annual → quarterly carry-forward)

Output:
  - data/cleaned/housing_model_quarterly_panel.csv
"""
import argparse, re
import pandas as pd
from pathlib import Path
from housing.config import CLEANED, RAW, PROCESSED

def to_qstr(s):
    # 把 period/datetime/字符串 统一成 'YYYYQn'
    try:
        return pd.PeriodIndex(s, freq="Q").astype(str)
    except Exception:
        return pd.to_datetime(s, errors="coerce").dt.to_period("Q").astype(str)

def read_hpi(path: Path):
    # 允许你之前生成的是“长表”
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if set(cols) >= {"quarter","city","index"}:
        long = df.copy()
        wide = long.pivot(index="quarter", columns="city", values="index").reset_index()
        wide.columns.name = None
        # 容忍 'sydney'/'melbourne' 大小写
        def pick(cols, key):
            for c in cols:
                if key.lower() in c.lower(): return c
            raise KeyError(f"Missing {key} in HPI columns: {cols}")
        syd = pick(wide.columns, "syd")
        mel = pick(wide.columns, "mel")
        out = wide[["quarter", syd, mel]].rename(columns={syd:"hpi_sydney", mel:"hpi_melbourne"})
    else:
        # 如果不小心传了“宽表”
        qcol = next((c for c in df.columns if c.lower()=="quarter"), df.columns[0])
        syd = next(c for c in df.columns if "syd" in c.lower())
        mel = next(c for c in df.columns if "mel" in c.lower())
        out = df[[qcol, syd, mel]].rename(columns={qcol:"quarter", syd:"hpi_sydney", mel:"hpi_melbourne"})
    out["quarter"] = to_qstr(out["quarter"])
    out = out.drop_duplicates("quarter").sort_values("quarter")
    return out

def read_population():
    p = CLEANED / "population_NSW_VIC_quarterly_2008_2025_wide.csv"
    df = pd.read_csv(p)
    df["quarter"] = to_qstr(df["quarter"])
    df = df.rename(columns={
        "NSW":"population_nsw", "VIC":"population_vic",
        "NSW_qoq_pct":"pop_qoq_nsw", "NSW_yoy_pct":"pop_yoy_nsw",
        "VIC_qoq_pct":"pop_qoq_vic", "VIC_yoy_pct":"pop_yoy_vic"
    })[["quarter","population_nsw","pop_qoq_nsw","pop_yoy_nsw",
        "population_vic","pop_qoq_vic","pop_yoy_vic"]]
    return df

def read_approvals():
    p = CLEANED / "approvals_quarterly_states_long_2008_onwards.csv"
    df = pd.read_csv(p)
    df["quarter"] = to_qstr(df["quarter"])
    # 只要“审批总量”指标
    df = df[df["metric"].str.contains("dwellings|approvals", case=False, na=False)]
    wide = df.pivot_table(index="quarter", columns="state", values="value", aggfunc="sum").reset_index()
    wide.columns.name = None
    return wide.rename(columns={"NSW":"approvals_units_nsw","VIC":"approvals_units_vic"})

def read_labour(state: str):
    fn = CLEANED / f"{state.lower()}_labour_quarterly_2008_onwards.csv"
    df = pd.read_csv(fn)
    df["quarter"] = to_qstr(df["quarter"])
    rename = {
        "emp_total": f"emp_total_{state.lower()}",
        "unemp_rate": f"unemp_rate_{state.lower()}",
        "part_rate": f"part_rate_{state.lower()}",
        "emp_qoq_pct": f"emp_qoq_pct_{state.lower()}",
        "emp_yoy_pct": f"emp_yoy_pct_{state.lower()}",
    }
    keep = ["quarter","emp_total","unemp_rate","part_rate","emp_qoq_pct","emp_yoy_pct"]
    return df[keep].rename(columns=rename)

def read_cash_quarterly():
    p = CLEANED / "cash_rate_quarterly.csv"
    df = pd.read_csv(p)
    # 兼容列名：quarter / quarter_end + quarter
    if "quarter" not in df.columns:
        if "quarter_end" in df.columns:
            df["quarter"] = pd.to_datetime(df["quarter_end"]).dt.to_period("Q-DEC").astype(str)
        else:
            raise KeyError("cash_rate_quarterly.csv missing 'quarter' or 'quarter_end'")
    df["quarter"] = to_qstr(df["quarter"])
    # 用季均（也可改为季末）
    return df[["quarter","cash_rate_avgq"]].rename(columns={"cash_rate_avgq":"cash_rate"})

def read_wealth_income():
    # 兼容两种位置
    candidate = [CLEANED/"wealth and income.csv", RAW/"wealth and income.csv"]
    p = next((x for x in candidate if x.exists()), None)
    if p is None:
        # 没有就返回空表
        return pd.DataFrame(columns=["quarter","avg_net_wealth_k","median_weekly_income_hilda"])
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    # 查找年份列
    year_col = next((c for c in df.columns if c.lower()=="year"), df.columns[0])
    df["Year"] = df[year_col].astype(int)
    # 猜测财富/收入列名（若已有正式列名则不会改）
    if "avg_net_wealth_k" not in df.columns and len(df.columns) >= 2:
        df = df.rename(columns={df.columns[1]:"avg_net_wealth_k"})
    if "median_weekly_income_hilda" not in df.columns and len(df.columns) >= 3:
        df = df.rename(columns={df.columns[2]:"median_weekly_income_hilda"})
    # 展开到季度（年内复制）
    calendar = pd.DataFrame({"quarter": pd.period_range("2008Q1","2025Q4", freq="Q").astype(str)})
    calendar["Year"] = calendar["quarter"].str[:4].astype(int)
    q = calendar.merge(df[["Year","avg_net_wealth_k","median_weekly_income_hilda"]],
                       on="Year", how="left").drop(columns="Year")
    return q

def clean_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                           .str.replace(",","", regex=False)
                           .str.replace(" ","", regex=False)
                           .str.strip())
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_panel():
    # 读取各表
    hpi = read_hpi(CLEANED / "house_price_index_SydMel_2008Q1_2024Q2.csv")
    pop = read_population()
    apv = read_approvals()
    lab_nsw = read_labour("NSW")
    lab_vic = read_labour("VIC")
    cash = read_cash_quarterly()
    wealth = read_wealth_income()

    # 拼“季度日历”为左表
    max_q = max(to_qstr(x["quarter"]).max() for x in [hpi,pop,apv,lab_nsw,lab_vic,cash,wealth])
    calendar = pd.DataFrame({"quarter": pd.period_range("2008Q1", max_q, freq="Q").astype(str)})

    df = (calendar
          .merge(hpi,      on="quarter", how="left")
          .merge(pop,      on="quarter", how="left")
          .merge(cash,     on="quarter", how="left")
          .merge(wealth,   on="quarter", how="left")
          .merge(apv,      on="quarter", how="left")
          .merge(lab_nsw,  on="quarter", how="left")
          .merge(lab_vic,  on="quarter", how="left"))

    # 数值清洗（去逗号/空格）
    num_cols = [
        "hpi_sydney","hpi_melbourne",
        "population_nsw","population_vic",
        "approvals_units_nsw","approvals_units_vic",
        "emp_total_nsw","unemp_rate_nsw","part_rate_nsw",
        "emp_total_vic","unemp_rate_vic","part_rate_vic",
        "cash_rate","avg_net_wealth_k","median_weekly_income_hilda"
    ]
    df = clean_numeric(df, [c for c in num_cols if c in df.columns])

    # 去重复季度
    df = df.drop_duplicates("quarter").sort_values("quarter").reset_index(drop=True)
    return df

def main(args):
    panel = build_panel()
    out = CLEANED / "housing_model_quarterly_panel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print("✅ Panel saved:", out)
    print("shape:", panel.shape, "| range:", panel['quarter'].min(), "→", panel['quarter'].max())
    # 打印各变量最后非空季度，方便检查缺口
    key = ["hpi_sydney","hpi_melbourne","cash_rate",
           "population_nsw","population_vic",
           "approvals_units_nsw","approvals_units_vic"]
    for k in key:
        if k in panel.columns:
            last = panel.loc[panel[k].notna(),"quarter"].max()
            print(f"  last non-null {k}: {last}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
