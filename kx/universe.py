"""U(t) per PREREG §2: TOPIX Core30 + Large70 + Mid400 as of t, with history and a bar on t."""
import pandas as pd

SCALE = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}
PRIME = "0111"  # J-Quants market segment code for Prime (used only by the fallback)


def universe(master: pd.DataFrame, close: pd.DataFrame, value: pd.DataFrame, t: pd.Timestamp,
             min_history: int = 60, fallback_n: int = 500) -> list[str]:
    """master: issues listed as of t (Code, ScaleCat, Mkt). close/value: date x code AdjC / Va panels."""
    t = pd.Timestamp(t)
    if master["ScaleCat"].isin(SCALE).any():
        codes = master.loc[master["ScaleCat"].isin(SCALE), "Code"]
    else:  # fallback: 500 Prime issues with the highest median trading value over the prior 60 days
        prime = master.loc[master["Mkt"].astype(str) == PRIME, "Code"]
        med = value.loc[:t].iloc[-61:-1].reindex(columns=prime).median()
        codes = med.dropna().nlargest(fallback_n).index.to_series()
    hist = close.loc[:t].reindex(columns=list(codes))
    ok = (hist.notna().sum() >= min_history) & hist.iloc[-1].notna() & (hist.index[-1] == t)
    return sorted(ok[ok].index)
