from pathlib import Path
import pandas as pd
import numpy as np


def _coerce_quarter_like(s: pd.Series) -> pd.PeriodIndex:
    """
    Coerce a string / datetime-like Series into a quarterly PeriodIndex.
    This is robust to minor formatting differences in the raw data.
    """
    try:
        return pd.PeriodIndex(
            s.astype(str).str.replace(r"\s+", "", regex=True),
            freq="Q"
        )
    except Exception:
        return pd.to_datetime(s, errors="coerce").dt.to_period("Q")


def load_long_or_wide(data_csv: Path) -> pd.DataFrame:
    """
    Load a panel of housing and macro data in either long or wide format,
    and return a unified long-format DataFrame with columns:

        region, quarter, hpi, (macro variables...), q_dt

    Accepted input formats
    ----------------------
    1) Long:
        state, quarter, rppi_index (or hpi) + macro variables
    2) Wide:
        hpi_SYD, hpi_MEL, ... + macro variables

    The function standardises the quarter variable and, where present,
    attaches a common set of macro variables, including:

        cash_rate, cash_rate_level, cash_rate_qavg, cash_rate_eoq,
        unemp_rate, part_rate, income, wealth, avg_net_wealth_k,
        approvals_units, population, completions_nat, completion_rate_nat,
        supply_gap, rate_x_supply_gap, nom_aus
    """
    df = pd.read_csv(data_csv, encoding="utf-8-sig", low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    # Standardise quarter column
    qcol = cols.get("quarter", "quarter")
    df[qcol] = _coerce_quarter_like(df[qcol])

    # Prefer long format if present
    has_state = "state" in cols
    has_rppi = "rppi_index" in cols or "hpi" in cols

    if has_state and has_rppi:
        region_col = cols.get("state")
        hpi_col = cols.get("rppi_index", cols.get("hpi"))
        df_long = df[[region_col, qcol, hpi_col]].rename(
            columns={region_col: "region", qcol: "quarter", hpi_col: "hpi"}
        ).copy()

        # Attach macro variables if present
        attach = [
            "cash_rate",
            "cash_rate_level",
            "cash_rate_qavg",
            "cash_rate_eoq",
            "unemp_rate",
            "part_rate",
            "income",
            "wealth",
            "avg_net_wealth_k",
            "approvals_units",
            "population",
            "completions_nat",
            "completion_rate_nat",
            "supply_gap",
            "rate_x_supply_gap",
            "nom_aus",
        ]
        for mc in attach:
            if mc in cols:
                df_long[mc] = df[cols[mc]]

    else:
        # Wide format: find all columns that start with "hpi_"
        hpi_keys = [c for c in df.columns if c.lower().startswith("hpi_")]
        if not hpi_keys:
            raise ValueError(
                "Expected long format (state + rppi_index) or wide format (hpi_*) "
                "but found neither."
            )

        frames = []
        for hk in hpi_keys:
            city = hk.replace("hpi_", "").upper()
            frames.append(
                pd.DataFrame(
                    {
                        "region": city,
                        "quarter": df[qcol],
                        "hpi": df[hk],
                    }
                )
            )
        df_long = pd.concat(frames, ignore_index=True)

        # Broadcast macro variables to each row
        attach = [
            "cash_rate",
            "cash_rate_level",
            "cash_rate_qavg",
            "cash_rate_eoq",
            "unemp_rate",
            "part_rate",
            "income",
            "wealth",
            "avg_net_wealth_k",
            "approvals_units",
            "population",
            "completions_nat",
            "completion_rate_nat",
            "supply_gap",
            "rate_x_supply_gap",
            "nom_aus",
        ]
        for mc in attach:
            if mc in cols:
                values = df[cols[mc]].values
                # Tile to match long frame length
                df_long[mc] = np.tile(values, len(df_long) // len(df))

    df_long = df_long.sort_values(["region", "quarter"]).reset_index(drop=True)
    df_long["q_dt"] = df_long["quarter"].dt.to_timestamp("Q")
    return df_long
