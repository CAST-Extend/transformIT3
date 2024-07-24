import os
import re
import pandas as pd
from sys import getsizeof
from dash import dash_table

def lsize():
    objs = {name: getsizeof(eval(name)) for name in globals()}
    df = pd.DataFrame(list(objs.items()), columns=['object', 'size'])
    df = df.sort_values(by='size', ascending=False).reset_index(drop=True)
    return df

def memcheck(env=globals()):
    objs = {name: getsizeof(eval(name, env)) for name in env}
    df = pd.DataFrame(list(objs.items()), columns=['object', 'size'])
    df = df.sort_values(by='size', ascending=False).reset_index(drop=True)
    return df

def my_datatable(data, selection="none"):
    table = dash_table.DataTable(
        data=data.to_dict('records'),
        columns=[{"name": i, "id": i} for i in data.columns],
        row_selectable=selection,
        filter_action="native",
        export_format="csv",
        export_headers="display",
        merge_duplicate_headers=True,
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_cell={'textAlign': 'left'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_as_list_view=True,
        column_selectable="multi",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current=0,
        page_size=10
    )
    df = pd.DataFrame(table.data)
    return df

look_for_case_change_regexp = r"(?!^)((?=[A-Z]{3,})(?<=[a-z])|(?=[A-Z][a-z]))"
look_for_position_before_space_regexp = r"(?=\s)"
look_for_position_after_space_regexp = r"(?<=\s)"
look_for_short_words_regexp = r"(?<=\s)(\w{1,2}\s)"
look_for_numbers_regexp = r"(?<=\s)(\d+\s)"
look_for_repeated_word = r"\b(\w+)(?:\W+\1\b)+"

PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
END = '\033[0m'
WARNING = '\033[93m'
FAIL = '\033[91m'
BOLD = '\033[1m'
ITALIC = '\x1B[3m'
UNDERLINE = '\033[4m'

def my_dirname_df_from_list(path_list):
    path_df = pd.DataFrame(path_list, columns=['path']).drop_duplicates().reset_index(drop=True)
    path_df['reversed'] = path_df['path'].apply(lambda x: x[::-1])
    path_df = path_df.assign(segment_idx=1).groupby('path').cumcount() + 1
    path_df = path_df.assign(eulav=path_df['reversed'].str.split(r'[/\\]').explode())
    path_df['segment_idx'] = path_df.groupby('path').cumcount() + 1
    path_df = path_df.sort_values(['path', 'segment_idx'], ascending=[True, False])
    
    path_dir = path_df.groupby('path').agg({'eulav': lambda x: '/'.join(x[::-1])}).reset_index()
    path_file = path_df.groupby('path').head(1).reset_index(drop=True)
    path_file['path_filename'] = path_file['eulav'].apply(lambda x: x[::-1])
    
    return pd.merge(path_dir, path_file, on='path')

def my_eu_from_df_file(discovery_file_path="C:/AppData/GitLab/exploration/RAKE/json_files/df_field__thingsboard.json"):
    import json
    
    with open(discovery_file_path, 'r') as file:
        data = json.load(file)
    
    execution_units = data['execution units']
    
    project_df = pd.json_normalize(execution_units, 'projects', ['path'])
    project_df['main_path'] = project_df['path']
    project_df = project_df[['name', 'description', 'main_path', 'path']].rename(columns={'path': 'other_path'})
    
    project_df['main_path'] = project_df['main_path'].str[4:].str.replace(r'\\', '/')
    project_df['other_path'] = project_df['other_path'].str[4:].str.replace(r'\\', '/')
    
    word_df = pd.json_normalize(execution_units, 'words', ['name'])
    word_df = word_df.groupby('name').agg({'word': list}).reset_index()
    
    project_df = project_df.groupby('name').agg({'description': 'first', 'main_path': 'first', 'other_path': list}).reset_index()
    result_df = pd.merge(project_df, word_df, on='name')
    result_df['service_name'] = result_df['name'].str.lower().str.replace(r'[^a-zA-Z0-9]', '_')
    result_df['description'] = result_df['description'].fillna('')
    
    return result_df
