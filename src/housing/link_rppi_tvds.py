import argparse
import pandas as pd
import re
from pathlib import Path
from housing.config import CLEANED
from housing.utils import to_quarter_label

def normalize_qstr(s: pd.Series) -> pd.Series:
    """
    超稳健季度标准化：
    1) 若本身是 'YYYYQn' 直接返回
    2) 否则清理不可见字符/破折号，尝试转 datetime -> Q-DEC
    3) 再兜底用正则从字符串里抓 YYYY 和 Qn
    """
    s = s.astype(str)\
         .str.replace("\u200b", "", regex=False)\
         .str.replace("\xa0", " ", regex=False)\
         .str.strip()

    # 1) 已是 'YYYYQn'
    direct = s.str.extract(r'^(?P<y>\d{4})\s*Q\s*(?P<q>[1-4])$', flags=re.I)
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    ok_direct = direct.notna().all(axis=1)
    out.loc[ok_direct] = direct.loc[ok_direct].apply(lambda r: f"{r.y}Q{r.q}", axis=1)

    # 2) datetime 解析
    rem = out.isna()
    if rem.any():
        s2 = s[rem].str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
        dt = pd.to_datetime(s2, errors="coerce")
        ok_dt = dt.notna()
        out.loc[rem & ok_dt] = dt[ok_dt].dt.to_period("Q-DEC").astype(str)

        # 3) 正则兜底
        rem2 = out.isna()
        if rem2.any():
            m = s2[rem2].str.extract(r'(?P<y>\d{4}).*?Q\s*(?P<q>[1-4])', flags=re.I)
            ok = m.notna().all(axis=1)
            out.loc[rem2 & ok] = m.loc[ok].apply(lambda r: f"{r.y}Q{r.q}", axis=1)

    return out.astype(str)

def read_rppi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    qcol = cols_lower.get("quarter", list(df.columns)[0])

    syd_col = next((c for c in df.columns if "syd" in c.lower()), None)
    mel_col = next((c for c in df.columns if "mel" in c.lower()), None)
    if syd_col is None or mel_col is None:
        raise KeyError(f"RPPI must contain Sydney/Melbourne columns. Got: {list(df.columns)}")

    out = df.rename(columns={qcol: "quarter"})[["quarter", syd_col, mel_col]]
    out.columns = ["quarter", "Sydney", "Melbourne"]
    out["quarter"] = normalize_qstr(out["quarter"])

    # 只保留有效 'YYYYQn'
    patt = r'^\d{4}Q[1-4]$'
    out = out[out["quarter"].str.match(patt, na=False)].copy()
    out = out.drop_duplicates("quarter").sort_values("quarter").reset_index(drop=True)
    return out

def find_header_row_xlsx(xlsx: Path, sheet, max_rows=200) -> int:
    tmp = pd.read_excel(xlsx, sheet_name=sheet, header=None, nrows=max_rows)
    for i in range(len(tmp)):
        row = " ".join([str(x) for x in tmp.iloc[i].tolist() if pd.notna(x)]).lower()
        if ("new south wales" in row) and ("victoria" in row):
            return i
    return 0

def read_tvds_wide(xlsx: Path, sheet="Data1") -> pd.DataFrame:
    hdr = find_header_row_xlsx(xlsx, sheet)
    df = pd.read_excel(xlsx, sheet_name=sheet, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]
    qcol = df.columns[0]

    def pick(name_key):
        for c in df.columns:
            if name_key.lower() == c.strip().lower():
                return c
        for c in df.columns:
            if name_key.lower() in c.strip().lower():
                return c
        raise KeyError(f"Cannot find column containing '{name_key}'. Have: {list(df.columns)}")

    col_nsw = pick("new south wales")
    col_vic = pick("victoria")
    out = df[[qcol, col_nsw, col_vic]].rename(columns={qcol: "quarter_raw", col_nsw: "NSW", col_vic: "VIC"})
    out["quarter"] = normalize_qstr(out["quarter_raw"])
    out = out.drop(columns=["quarter_raw"]).dropna(subset=["quarter"])\
             .drop_duplicates("quarter").sort_values("quarter").reset_index(drop=True)
    return out

def to_index(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    base = x.iloc[0]
    return x / base * 100.0

def link_series(rppi: pd.DataFrame, tvds: pd.DataFrame, start_q: str, end_q: str):
    rppi = rppi[(rppi["quarter"] >= start_q) & (rppi["quarter"] <= end_q)].copy()
    tvds = tvds[(tvds["quarter"] >= "2011Q1") & (tvds["quarter"] <= end_q)].copy()

    tvds_idx = tvds.copy()
    tvds_idx["Sydney_proxy"]    = to_index(tvds_idx["NSW"])
    tvds_idx["Melbourne_proxy"] = to_index(tvds_idx["VIC"])
    tvds_idx = tvds_idx[["quarter", "Sydney_proxy", "Melbourne_proxy"]]

    overlap = pd.merge(rppi[["quarter", "Sydney", "Melbourne"]], tvds_idx, on="quarter", how="inner")
    if overlap.empty:
        raise RuntimeError(
            "No overlap between RPPI and TVDS. "
            f"RPPI range {rppi['quarter'].min()}→{rppi['quarter'].max()}, "
            f"TVDS range {tvds_idx['quarter'].min()}→{tvds_idx['quarter'].max()}."
        )

    last = overlap.iloc[-1]
    s_scale = last["Sydney"]    / last["Sydney_proxy"]
    m_scale = last["Melbourne"] / last["Melbourne_proxy"]

    tvds_scaled = tvds_idx.copy()
    tvds_scaled["Sydney"]    = tvds_scaled["Sydney_proxy"]    * s_scale
    tvds_scaled["Melbourne"] = tvds_scaled["Melbourne_proxy"] * m_scale
    tvds_scaled = tvds_scaled[["quarter", "Sydney", "Melbourne"]]

    cut_q = last["quarter"]
    part1 = rppi[rppi["quarter"] <= cut_q]
    part2 = tvds_scaled[tvds_scaled["quarter"] > cut_q]
    linked = pd.concat([part1, part2], ignore_index=True)\
               .drop_duplicates("quarter").sort_values("quarter").reset_index(drop=True)

    long = linked.melt(id_vars="quarter", var_name="city", value_name="index")
    return long, cut_q

def main(args):
    rppi = read_rppi(Path(args.rppi))
    tvds = read_tvds_wide(Path(args.tvds), sheet=args.sheet)
    long, cut_q = link_series(rppi, tvds, args.start, args.end)
    out = CLEANED / f"house_price_index_SydMel_{args.start}_{args.end}.csv"
    long.to_csv(out, index=False)
    print("OK Linked RPPI + TVDS")
    print(f"overlap cut @ {cut_q}")
    print(f"saved: {out}")
    print(f"rows: {len(long)} | range: {long['quarter'].min()} → {long['quarter'].max()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rppi",  required=True, help="RPPI wide CSV (Sydney/Melbourne)")
    ap.add_argument("--tvds",  required=True, help="ABS TVDS Excel (e.g., 643201.xlsx)")
    ap.add_argument("--sheet", default="Data1")
    ap.add_argument("--start", default="2008Q1")
    ap.add_argument("--end",   default="2024Q2")
    main(ap.parse_args())
