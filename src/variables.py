from dataclasses import dataclass


@dataclass(frozen=True)
class Var:
    loan_id: str = 'id'
    train_period_from_year: int = 2013
    train_period_to_year: int = 2017
    test_period_from_year: int = 2018
    test_period_to_year: int = 2018
    # only exist in the 2007-2018Q4 version
    member_id: str = 'member_id'
    time_id: str = 'issue_d'
    target: str = 'loan_status'
    last_payment: str = 'last_pymnt_d'

    numeric_features: tuple = (
        'loan_amnt',
        # main borrower
        'annual_inc',
        'dti',
        'emp_length',
        'fico_range_high',
        'fico_range_low',
        # co borrower
        'annual_inc_joint',
        'dti_joint',
        'sec_app_fico_range_low',
        'sec_app_fico_range_high',
    )

    categorical_features: tuple = (
        'emp_title',
        'addr_state',
        'application_type',
        # 'desc',
        'home_ownership',
        'policy_code',
        'purpose',
        'zip_code'
    )

    date_features: tuple = (
        'issue_d',
        'earliest_cr_line',
        'sec_app_earliest_cr_line',
    )