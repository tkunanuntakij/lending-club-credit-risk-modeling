from pathlib import Path
from src.variables import Var
import pandas as pd


def process_raw_data(path: Path, limit: int|None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        index_col=[Var.loan_id],
        usecols=[
            Var.time_id,
            Var.loan_id,
            Var.target,
            Var.last_payment,
            *Var.numeric_features,
            *Var.categorical_features,
            *Var.date_features,
        ],
        parse_dates=[
            Var.time_id,
            Var.last_payment,
            *Var.date_features
        ],
        date_format="%b-%Y",
        dtype={
            "emp_length": "string"
        },
        low_memory=False,
        nrows=limit
    )
    years = df[Var.time_id].dt.year
    df = df[(years >= Var.train_period_from_year) & (years <= Var.test_period_to_year)]
    for col in Var.date_features:
        df[col] = df[col].dt.to_period("M")
    df[Var.last_payment] = df[Var.last_payment].dt.to_period("M")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = df[Var.time_id].dt.year
    train = df[(years >= Var.train_period_from_year) & (years <= Var.train_period_to_year)]
    test = df[(years >= Var.test_period_from_year) & (years <= Var.test_period_to_year)]
    return (train, test)


def clean_data(df: pd.DataFrame):
    pass
