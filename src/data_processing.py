import re
from pathlib import Path
from src.variables import Var
import pandas as pd
import numpy as np


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


def drop_unused_variables(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(Var.unused_features), errors='ignore')


def handle_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[Var.target].isin([
        'Fully Paid',
        'Charged Off',
        'Late (31-120 days)',
        'Default'
    ])]

    df['default'] = df[Var.target].isin([
        'Charged Off',
        'Late (31-120 days)',
        'Default'
    ]).astype(int)
    df = df.drop(columns=Var.target)
    return df


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    df['dti'] = df['dti'].fillna(0)
    df['annual_inc'] = df['annual_inc'].fillna(0)
    return df


def drop_duplicated_variables(df: pd.DataFrame) -> pd.DataFrame:
    if 'fico_range_high' in df.columns:
        df = df.drop(columns='fico_range_high').rename(columns={'fico_range_low': 'fico'})
    if 'sec_app_fico_range_high' in df.columns:
        df = df.drop(columns='sec_app_fico_range_high').rename(columns={'sec_app_fico_range_low': 'sec_fico'})
    return df


def map_emp_length(value: str) -> int:
    if len(value) == 0:
        return 0
    replaced_value = re.sub(r'\s+year[s]?', '', value)
    if replaced_value is None:
        return 0
    if replaced_value == '< 1':
        return 1
    elif replaced_value == '10+':
        return 10
    else:
        return int(replaced_value)


def handle_employment_years(df: pd.DataFrame) -> pd.DataFrame:
    numeric_emp_length = df['emp_length'].fillna('').map(map_emp_length)
    numeric_emp_length = numeric_emp_length.fillna(0)
    df['emp_length'] = numeric_emp_length
    return df


def merge_joint_loan_info(df: pd.DataFrame) -> pd.DataFrame:
    joint_loan_mask = df['application_type'] == 'Joint App'
    joint_df = df[joint_loan_mask]
    
    # handle DTI
    debt = joint_df['dti'] * joint_df['annual_inc']
    joint_debt = joint_df['dti_joint'] * joint_df['annual_inc_joint'].fillna(0)
    df.loc[joint_loan_mask, 'dti'] = (debt + joint_debt) / (joint_df['annual_inc'] + joint_df['annual_inc_joint'].fillna(0))
    df.loc[joint_loan_mask, 'dti'] = (debt + joint_debt) / (joint_df['annual_inc'] + joint_df['annual_inc_joint'].fillna(0))
    
    # handle income
    # The are a couple of approach possible.
    # e.g. (1) Use total income (2) Use the higher one (3) Use the lower one (4) Use the average
    # joint loan should be stronger than a sole loan from each applicant,
    # but it may not be as strong as a sole loan of an applicant who earn the total income.
    # I guess, I will use (2) to be conservative.
    df.loc[joint_loan_mask, 'annual_inc'] = np.max([joint_df['annual_inc'], joint_df['annual_inc_joint']], axis=0)
    
    # handle fico
    df.loc[joint_loan_mask, 'fico'] = np.max([joint_df['fico'], joint_df['sec_fico']], axis=0)

    # handle loan experience
    df.loc[joint_loan_mask, 'loan_exp'] = np.max([joint_df['loan_exp'], joint_df['loan_exp_joint']], axis=0)

    df = df.drop(columns=['dti_joint', 'annual_inc_joint', 'sec_fico', 'loan_exp_joint'])
    return df


def determine_loan_experience(df: pd.DataFrame) -> pd.DataFrame:
    df['loan_exp'] = df[Var.time_id].dt.start_time - df['earliest_cr_line'].dt.start_time
    df['loan_exp_joint'] = df[Var.time_id].dt.start_time - df['sec_app_earliest_cr_line'].dt.start_time
    df = df.drop(columns=['earliest_cr_line', 'sec_app_earliest_cr_line'])
    return df

def clean_data(df: pd.DataFrame):
    df = (
        df.pipe(drop_unused_variables)
        .pipe(fill_na)
        .pipe(handle_loan_status)
        .pipe(determine_loan_experience)
        .pipe(drop_duplicated_variables)
        .pipe(handle_employment_years)
        .pipe(merge_joint_loan_info)
    )
    return df
