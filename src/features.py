import numpy as np
import pandas as pd


def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log, difference and interaction transforms used in the models.

    The goal is to keep the modelling code simple by centralising all
    feature engineering here.
    """
    d = df.copy()

    # Log house price index
    d["ln_hpi"] = np.log(d["hpi"])

    # Basic log transforms for core levels (if available)
    for c in ["population", "approvals_units"]:
        if c in d.columns:
            d[f"ln_{c}"] = np.log(d[c].replace({0: np.nan}))

    # First differences by region for selected variables
    d["d_ln_hpi"] = d.groupby("region")["ln_hpi"].diff()

    if "cash_rate_level" in d.columns:
        d["d_cash_rate_level"] = d["cash_rate_level"].diff()

    if "ln_approvals_units" not in d.columns and "approvals_units" in d.columns:
        d["ln_approvals_units"] = np.log(d["approvals_units"].replace({0: np.nan}))

    if "ln_population" not in d.columns and "population" in d.columns:
        d["ln_population"] = np.log(d["population"].replace({0: np.nan}))

    for c in ["ln_approvals_units", "ln_population", "nom_aus"]:
        if c in d.columns:
            d[f"d_{c}"] = d.groupby("region")[c].diff()

    # Interaction: interest rate × supply gap
    if "cash_rate_level" in d.columns and "supply_gap" in d.columns:
        d["rate_x_supply_gap"] = d["cash_rate_level"] * d["supply_gap"]

    return d
