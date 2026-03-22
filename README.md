# Credit Risk Modeling on Lending Club Dataset

## Overview


## Dataset

- (Lending Club 2008-2020Q1)[https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1]

Download the dataset via Kaggle commandline tools

```bash
kaggle datasets download ethon0426/lending-club-20072020q1
```


## Preprocessing

- Information from the application only
- time period
    - 2013–2017 Training set Stable policy, sufficient maturity, clean outcomes
    - 2018 OOT validation Clean vintage, no COVID, sufficient maturity by end-2019

- combine info for loan with co-borrower
- convert data into parquet
