"""
Robust RPPI reader: extracts Sydney & Melbourne quarterly series (2008+)
- 自动判断 Excel/CSV
- CSV 自动嗅探分隔符（逗号/分号/制表）
- 自动定位含有 'Sydney' & 'Melbourne' 的表头行
- 解析各种季度写法到 'YYYYQn'

Run:
  python -m housing.rppi_reader --input "data/raw/Residential Property Price Indexes, capital cities.csv"
"""
import argparse, re
import pandas as pd
from pathlib import Path
from housing.config import CLEANED
from housing.utils import sniff_encoding

MONTHS = {m.lower(): i for i,m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

def is_excel(path: Path) -> bool:
    if path.suffix.lower() in [".xlsx",".xls"]: 
        return True
    # 某些“CSV”其实是Excel打出来的；用签名判断
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        return head.startswith(b"PK\x03\x04")  # xlsx zip header
    except Exception:
        return False

def sniff_sep(path: Path, encoding: str) -> str:
    # 简易嗅探：统计前200行中 , ; \t 的出现次数
    cnt = {",":0, ";":0, "\t":0}
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 200: break
            for s in cnt:
                cnt[s] += line.count(s)
    # 选最多的；若全部很少，返回 None 让 pandas 自动推断
    best = max(cnt, key=cnt.get)
    return best if cnt[best] > 0 else None

def find_header_row_text(path: Path, encoding: str, sep: str|None) -> int:
    # 为了稳健，不依赖 pandas 解析前，先用纯文本找表头
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        lines = [next(f, "") for _ in range(300)]
    for i, line in enumerate(lines):
        low = line.lower()
        if "sydney" in low and "melbourne" in low:
            return i
    return 0

def find_header_row_excel(path: Path, sheet: str|int|None=None) -> int:
    tmp = pd.read_excel(path, sheet_name=sheet or 0, header=None, nrows=300)
    for i in range(len(tmp)):
        row = " ".join(str(x) for x in tmp.iloc[i].dropna().tolist()).lower()
        if "sydney" in row and "melbourne" in row:
            return i
    return 0

def parse_quarter_text(s: str|None):
    """Return 'YYYYQn' from many variants."""
    if s is None: return None
    t = str(s).strip().replace("\u200b","").replace("\xa0"," ").replace("–","-").replace("—","-")
    if not t: return None
    m = re.fullmatch(r"(\d{4})\s*Q\s*([1-4])", t, flags=re.I)
    if m: return f"{m.group(1)}Q{m.group(2)}"
    m = re.search(r"(?P<mon>[A-Za-z]{3})[\s\-]*(?P<y>\d{2,4})", t)
    if m:
        y = int(m.group("y")); y = 2000+y if y<100 else y
        mon = MONTHS.get(m.group("mon").lower())
        if mon: return f"{y}Q{(mon-1)//3+1}"
    m = re.search(r"(?P<y>\d{4})[\s\-]*(?P<mon>[A-Za-z]{3})", t)
    if m:
        y = int(m.group("y")); mon = MONTHS.get(m.group("mon").lower())
        if mon: return f"{y}Q{(mon-1)//3+1}"
    dt = pd.to_datetime(t, errors="coerce")
    if pd.notna(dt): return f"{dt.year}Q{(dt.month-1)//3+1}"
    return None

def main(args):
    raw = Path(args.input)
    if not raw.exists(): raise FileNotFoundError(raw)

    if is_excel(raw):
        hdr = find_header_row_excel(raw, sheet=args.sheet)
        df  = pd.read_excel(raw, sheet_name=args.sheet or 0, header=hdr)
    else:
        enc = sniff_encoding(raw)
        sep = sniff_sep(raw, enc)  # 逗号/分号/制表
        hdr = find_header_row_text(raw, enc, sep)
        # sep=None 让 pandas 自动推断；否则用嗅探分隔符
        df = pd.read_csv(raw, header=hdr, sep=sep, engine="python", encoding=enc)

    df.columns = [str(c).strip() for c in df.columns]
    qcol = df.columns[0]

    # 找城市列（容忍大小写/额外注释）
    def pick(cols, kw):
        for c in cols:
            if kw.lower() in c.lower():
                return c
        raise KeyError(f"Missing column containing '{kw}'. Columns: {list(cols)}")
    s_col = pick(df.columns, "sydney")
    m_col = pick(df.columns, "melbourne")

    sub = df[[qcol, s_col, m_col]].rename(columns={qcol:"q_raw", s_col:"Sydney", m_col:"Melbourne"})
    sub["quarter"] = sub["q_raw"].map(parse_quarter_text)

    wide = (sub.drop(columns=["q_raw"])
                .dropna(subset=["quarter"])
                .drop_duplicates("quarter")
                .sort_values("quarter")[["quarter","Sydney","Melbourne"]]
                .reset_index(drop=True))

    if len(wide) < 10:
        raise RuntimeError(f"Parsed only {len(wide)} rows. Please check source file format/columns.")

    long = wide.melt(id_vars="quarter", var_name="city", value_name="rppi_index")\
               .sort_values(["city","quarter"])

    out_wide = CLEANED / "rppi_syd_mel_quarterly_wide.csv"
    out_long = CLEANED / "rppi_syd_mel_quarterly_long.csv"
    wide.to_csv(out_wide, index=False)
    long.to_csv(out_long, index=False)

    print("✅ RPPI extracted")
    print(f"  rows: {len(wide)} | range: {wide['quarter'].iloc[0]} → {wide['quarter'].iloc[-1]}")
    print(f"  saved: {out_wide}\n         {out_long}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to RPPI capital-cities file (CSV or XLSX)")
    ap.add_argument("--sheet", help="Excel sheet name or index (if Excel)", default=None)
    main(ap.parse_args())
