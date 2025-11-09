import pandas as pd

def to_quarter_label(x, q="Q-DEC"):
    s = pd.to_datetime(pd.Series(x), errors="coerce")
    if s.notna().any():
        return s.dt.to_period(q).astype(str)
    try:
        return pd.PeriodIndex(pd.Series(x), freq="Q").astype(str)
    except Exception:
        return pd.Series(x).astype(str)

def sniff_encoding(path):
    for enc in ["utf-8-sig","utf-16","utf-16-le","utf-16-be","utf-8","ISO-8859-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                f.readline()
            return enc
        except Exception:
            pass
    return "utf-8"
