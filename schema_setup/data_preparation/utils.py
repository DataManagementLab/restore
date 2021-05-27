import csv
import os
import pickle

import pandas as pd


def read_table_csv(csv_file_location, column_names, drop_columns, csv_seperator, header=None, ensure_numeric=None):
    """
    Reads csv from path, renames columns and drops unnecessary columns
    """
    df_rows = pd.read_csv(csv_file_location, escapechar='\\', encoding='utf-8', quotechar='"', sep=csv_seperator,
                          error_bad_lines=False)
    df_rows.columns = column_names
    if ensure_numeric is not None:
        for col in ensure_numeric:
            df_rows = df_rows[pd.to_numeric(df_rows[col], errors='coerce').notnull()]

    for attribute in drop_columns:
        df_rows.drop(attribute, axis=1, inplace=True)

    return df_rows.apply(pd.to_numeric, errors="ignore")


def load_pkl(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def save_pkl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_csv(csv_rows, target_csv_path):
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)

    with open(target_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)


def batch(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]
