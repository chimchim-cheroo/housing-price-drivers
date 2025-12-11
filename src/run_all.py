from pathlib import Path

from data_io import load_long_or_wide
from features import add_transforms
from models import run_fe, run_dl, run_ardl
from figures import plot_ardl_short_long, plot_dl_cumulative, plot_fe_coeffs


DATA_CANDIDATES = [
    Path("data/processed/panel_wide.csv"),
    Path("data/cleaned/panel_wide.csv"),
    Path("data/cleaned/panel_long.csv"),
]
OUT_DIR = Path("outputs")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_csv = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_csv is None:
        raise FileNotFoundError(
            "No input panel CSV found. Please place a file at "
            "data/processed/panel_wide.csv or data/cleaned/panel_wide.csv "
            "or data/cleaned/panel_long.csv."
        )

    df_long = load_long_or_wide(data_csv)
    df_long = add_transforms(df_long)

    fe, base_X = run_fe(df_long, OUT_DIR)
    run_dl(df_long, OUT_DIR, vars_=base_X)
    run_ardl(df_long, OUT_DIR, vars_=base_X)

    # Figures
    plot_ardl_short_long(
        OUT_DIR / "ARDL_effects_summary.csv",
        OUT_DIR / "FIG_ARDL_short_long.png",
    )
    plot_dl_cumulative(
        OUT_DIR / "DL_longrun_effects.csv",
        OUT_DIR / "FIG_DL_cumulative.png",
    )
    plot_fe_coeffs(
        OUT_DIR / "fe_coeffs.csv",
        OUT_DIR / "FIG_FE_coeffs.png",
    )

    print("Pipeline completed. Outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
