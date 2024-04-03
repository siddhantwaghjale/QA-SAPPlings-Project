import requests
import os
import pandas as pd
import numpy as np

import json
import pickle


def save_new_chain(user_query, output_chain, answer):
    try:
        df = pd.read_csv('./data/chains.csv', index_col=0)
        df.loc[len(df)] = {'query':user_query, 'chain':output_chain, 'answer': answer}
        df.to_csv('./data/chains.csv')

    except FileNotFoundError:
        df = pd.DataFrame([[user_query, output_chain, answer]])
        df.columns = ['query', 'chain', 'answer']
        df.to_csv('./data/chains.csv')

def get_last_chains(how_many=5):
    try:
        df = pd.read_csv('./data/chains.csv', index_col=0)
        return df.tail(how_many)
    except:
        return ''


def clean_table(table):
    table = table.astype(str)
    table.fillna("", inplace=True)
    if len(table) > 0:
        # rename all duplicate column names
        if len(set(table.columns.to_list())) != len(table.columns):
            column_name_indices = defaultdict(list)
            for i, col_name in enumerate(table.columns):
                column_name_indices[col_name].append(i)
            new_column_names = [''] * len(table.columns)
            for col_name, indices in column_name_indices.items():
                if len(indices) > 1:
                    for j, index in enumerate(indices):
                        new_column_names[index] = f"{col_name} -{j + 1}"
                        # table.rename(inplace=True,columns={f'{index+1}col':f"{col_name}-{j+1}"})
                else:
                    new_column_names[indices[0]] = col_name
            table.columns = new_column_names

        # check all values of table and map them to str
        table = table.map(
            lambda x: pd.to_datetime(x, infer_datetime_format=True).strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(x, (pd.Timestamp, np.datetime64)) else x)
        table = table.map(lambda x: str(x))
        table.columns = table.columns.astype(str)
    return table