# Credit Risk Modeling on Lending Club Dataset

## Overview


## Dataset

2007_to_2018Q4
https://www.kaggle.com/datasets/wordsforthewise/lending-club

2007_to_2020Q3
https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1?select=Loan_status_2007-2020Q3.gzip


## Preprocessing

- Information from the application only
- time period
    - 2013–2017 Training set Stable policy, sufficient maturity, clean outcomes
    - 2018 OOT validation Clean vintage, no COVID, sufficient maturity by end-2019

- combine info for loan with co-borrower
- convert data into parquet