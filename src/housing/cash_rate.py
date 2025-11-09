"""
RBA Cash Rate pipeline
Input: RBA cash rate history (CSV/XLSX), messy headers tolerated
Outputs (to data/cleaned/):
  - cash_rate_announcements.csv   (announcement date-level)
  - cash_rate_daily.csv           (daily step function, ffill)
  - cash_rate_monthly.csv         (end-of-month & monthly average)
  - cash_rate_quarterly.csv       (end-of-quarter & quarterly average)

Run:
  python -m housing.cash_rate --input data/raw/cash_ratehist.xlsx --sheet Data
"""
import argparse, re
import pandas as pd
from pathlib import Path
from housing.config import CLEANED

def is_excel(path: Path) -> bool:
    return path.suffix.lower() in [".xlsx",".xls"]

def find_header_row_excel(path: Path, sheet=None, max_rows=60) -> int:
    tmp = pd.read_excel(path, sheet_name=sheet or 0, header=None, nrows=max_rows)
    for i in range(len(tmp)):
        row = " | ".join(str(x) for x in tmp.iloc[i].tolist() if pd.notna(x)).lower()
        if "new cash rate target" in row or "cash rate" in row:
            return i
    return 0

def parse_cash_value(x):
    """Parse 'New Cash Rate Target' text to float percent, tolerant of 'a to b'."""
    if pd.isna(x): return pd.NA
    t = str(x).strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|–|-)\s*(-?\d+(?:\.\d+)?)", t, flags=re.I)
    if m: return float(m.group(2))
    m = re.search(r"(-?\d+(?:\.\d+)?)", t)
    if m: return float(m.group(1))
    return pd.NA

def read_cash_table(path: Path, sheet=None) -> pd.DataFrame:
    if is_excel(path):
        hdr = find_header_row_excel(path, sheet)
        df = pd.read_excel(path, sheet_name=sheet or 0, header=hdr)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    # 猜日期列（'date'/'publication date' 等）
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    # 猜目标列
    rate_col = next((c for c in df.columns if "new cash rate target" in c.lower()), None)
    if rate_col is None:
        rate_col = next((c for c in df.columns if "cash rate" in c.lower()), df.columns[1])

    out = df[[date_col, rate_col]].rename(columns={date_col:"date_raw", rate_col:"cash_raw"})
    out["date"] = pd.to_datetime(out["date_raw"], errors="coerce")
    out["cash_rate"] = out["cash_raw"].apply(parse_cash_value)
    out = (out.dropna(subset=["date","cash_rate"])
               .sort_values("date")
               .drop_duplicates("date"))
    return out[["date","cash_rate"]]

def build_series(df: pd.DataFrame):
    # 公告级
    ann = df.copy()
    ann["cash_rate_change"] = ann["cash_rate"].diff()

    # 日度阶梯（前向填充到 2025-12-31）
    daily = ann.set_index("date")[["cash_rate"]]
    full_days = pd.DataFrame(index=pd.date_range(daily.index.min(), "2025-12-31", freq="D"))
    daily = full_days.join(daily).ffill()

    # 月度：月末末值 / 月均
    monthly = pd.DataFrame({
        "cash_rate_eom": daily["cash_rate"].resample("M").last(),
        "cash_rate_avgm": daily["cash_rate"].resample("M").mean()
    }).reset_index(names="month")

    # 季度：季末末值 / 季均（Q-DEC）
    quarterly = pd.DataFrame({
        "cash_rate_eoq": daily["cash_rate"].resample("Q-DEC").last(),
        "cash_rate_avgq": daily["cash_rate"].resample("Q-DEC").mean()
    }).reset_index(names="quarter_end")
    quarterly["quarter"] = quarterly["quarter_end"].dt.to_period("Q-DEC").astype(str)
    quarterly = quarterly[["quarter","cash_rate_eoq","cash_rate_avgq"]]

    return ann, daily.reset_index(), monthly, quarterly

def main(args):
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)

    df = read_cash_table(src, sheet=args.sheet)
    df = df[(df["date"] >= "2008-01-01") & (df["date"] <= "2025-12-31")]

    ann, daily, monthly, quarterly = build_series(df)

    (CLEANED / "cash_rate_announcements.csv").write_text(
        ann.to_csv(index=False)
    )
    daily.to_csv(CLEANED / "cash_rate_daily.csv", index=False)
    monthly.to_csv(CLEANED / "cash_rate_monthly.csv", index=False)
    quarterly.to_csv(CLEANED / "cash_rate_quarterly.csv", index=False)

    print("✅ Cash rate outputs saved to data/cleaned/:")
    print("  - cash_rate_announcements.csv")
    print("  - cash_rate_daily.csv")
    print("  - cash_rate_monthly.csv")
    print("  - cash_rate_quarterly.csv")
    print(f"Quarterly range: {quarterly['quarter'].min()} → {quarterly['quarter'].max()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="RBA cash rate history file (xlsx/csv)")
    ap.add_argument("--sheet", default=None, help="Excel sheet name/index if applicable")
    main(ap.parse_args())
