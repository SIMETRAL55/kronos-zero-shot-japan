import pandas as pd
import pytest

from kx.kronos_env import VENDOR, load_kronos

COLS = ["open", "high", "low", "close", "volume", "amount"]


@pytest.fixture(scope="session")
def kronos():
    return load_kronos()


@pytest.fixture(scope="session")
def reg_df():
    return pd.read_csv(VENDOR / "tests/data/regression_input.csv", parse_dates=["timestamps"])


def window(df: pd.DataFrame, start: int):
    """40 input bars from `start`, plus the timestamps of the 12 bars after them."""
    x = df.iloc[start:start + 40]
    y_ts = pd.DatetimeIndex(df["timestamps"].iloc[start + 40:start + 52])
    return x[COLS].reset_index(drop=True), pd.DatetimeIndex(x["timestamps"]), y_ts
