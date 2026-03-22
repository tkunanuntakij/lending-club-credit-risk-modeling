import sys
from pathlib import Path


raw_datafile = 'data/raw/Loan_status_2007_2020Q3.csv'
processed_train_output = 'data/processed/loan_train.parquet'
processed_test_output = 'data/processed/loan_test.parquet'
project_root = Path(__file__).parent.parent
datafile = Path(raw_datafile)

sys.path.append(project_root.as_posix())
from src.data_processing import process_raw_data, split_data


if __name__ == '__main__':
    df = process_raw_data(project_root / datafile)
    train_df, test_df = split_data(df)
    train_df.to_parquet(project_root / processed_train_output)
    test_df.to_parquet(project_root / processed_test_output)
    print(
        f"Finish processing: Train dataframe with dimension {train_df.shape}"
        + f"and test dataframe with dimension {test_df.shape}."
    )
