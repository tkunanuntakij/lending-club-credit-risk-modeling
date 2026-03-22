import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes,
        "missing": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "unique": df.nunique(),
        "sample": df.iloc[0],
    })
