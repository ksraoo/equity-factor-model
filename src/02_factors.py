"""
Step 2: Build momentum factors and check whether they relate to forward returns.

Key rule for every factor in this project: the value on date t may only use
information available on or before date t. Anything else is lookahead bias
and silently ruins the backtest.
"""

import pandas as pd
import numpy as np

PANEL_PATH = "data/price_panel.parquet"
OUTPUT_PATH = "data/factor_panel.parquet"

TRADING_DAYS_PER_MONTH = 21


def add_momentum(panel):
    """
    Momentum = how much the stock has risen over a past window.

    mom_3m, mom_6m: plain trailing returns.
    mom_12_1: 12-month return that SKIPS the most recent month. Academic
    research finds the last month tends to reverse, so the standard version
    excludes it.

    Worked example for mom_12_1 on one date t:
        price 21 days ago  / price 252 days ago  - 1
    Both prices are in the past, so no future information leaks in.
    """
    panel = panel.sort_values(["ticker", "date"]).copy()
    g = panel.groupby("ticker")["close"]

    m = TRADING_DAYS_PER_MONTH
    panel["mom_3m"] = g.transform(lambda x: x / x.shift(3 * m) - 1)
    panel["mom_6m"] = g.transform(lambda x: x / x.shift(6 * m) - 1)
    panel["mom_12_1"] = g.transform(lambda x: x.shift(1 * m) / x.shift(12 * m) - 1)
    return panel


def add_cross_sectional_zscore(panel, cols):
    """
    Standardize each factor WITHIN each date, across stocks.

    We care about which stocks look strong relative to the others on the same
    day, not the raw level. A z-score of +1 means "one standard deviation
    above the average stock today".
    """
    for col in cols:
        grp = panel.groupby("date")[col]
        panel[col + "_z"] = (panel[col] - grp.transform("mean")) / grp.transform("std")
    return panel


def information_coefficient(panel, factor_col, target_col="fwd_return"):
    """
    IC = on each date, the rank correlation between the factor and the
    forward return across stocks. Then we look at its average over time.

    Positive average IC means: stocks the factor ranked higher tended to
    earn higher forward returns.
    """
    def _ic(df):
        d = df[[factor_col, target_col]].dropna()
        if len(d) < 5:
            return np.nan
        return d[factor_col].corr(d[target_col], method="spearman")

    ic_by_date = panel.groupby("date").apply(_ic, include_groups=False).dropna()
    return ic_by_date


if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_PATH)
    panel = add_momentum(panel)

    factor_cols = ["mom_3m", "mom_6m", "mom_12_1"]
    panel = add_cross_sectional_zscore(panel, factor_cols)

    print("Factor NaN counts (expected: leading rows lost to the lookback window):")
    print(panel[factor_cols].isna().sum())

    print("\nInformation coefficient summary:")
    for col in factor_cols:
        ic = information_coefficient(panel, col + "_z")
        t_stat = ic.mean() / (ic.std() / np.sqrt(len(ic)))
        print(f"{col}: mean IC = {ic.mean():.4f}, IC std = {ic.std():.4f}, "
              f"t-stat = {t_stat:.2f}, n_dates = {len(ic)}")

    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
