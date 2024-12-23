import pandas as pd

from src.get_column_names import get_column_names


def imputation_data(dataset, type_predict):
    df = dataset
    column_names = get_column_names(df)
    for column_name in column_names:
        df[column_name].fillna(df[column_name].mode()[0], inplace=True)
        df[column_name] = df[column_name].apply(str)
    df.isna().sum().sort_values()
    df = df.drop_duplicates()
    df.isna().sum().sort_values()

    columns_to_drop = ['_id', 'landing_page_group_id', 'landing_page_id', 'offer_id']
    updated_series = pd.Series(column_names)
    updated_series = updated_series[~updated_series.isin(columns_to_drop)]
    column_names_x = updated_series.tolist()
    data_columns = column_names_x

    x = df[data_columns]
    y = df[[type_predict]]
    total_rows = len(df.axes[0])
    return data_columns, total_rows, x, y
