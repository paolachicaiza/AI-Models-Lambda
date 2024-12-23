import pandas as pd


def get_column_names(dataset):
    if isinstance(dataset, pd.DataFrame):
        return list(dataset.columns)
    else:
        raise ValueError("Invalid dataset format. Expected a Pandas DataFrame.")
