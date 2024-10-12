from typing import Optional, List, Union
import pandas as pd
import json


def assignee_to_dataframe(assignee_data: Optional[Union[dict, list]]) -> pd.DataFrame:
    if not assignee_data or pd.isna(assignee_data):
        return pd.DataFrame()

    try:
        if isinstance(assignee_data, dict):
            data = [assignee_data]
        elif isinstance(assignee_data, list):
            data = assignee_data
        else:
            return pd.DataFrame()

        assignee_df = pd.DataFrame(data)
        return assignee_df

    except Exception as e:
        return pd.DataFrame()


def assignees_to_dataframe_list(assignees_data: Optional[Union[dict, list]]) -> List[pd.DataFrame]:
    if not assignees_data or pd.isna(assignees_data):
        return []

    if isinstance(assignees_data, list):

        assignee_dfs = [assignee_to_dataframe(assignee) for assignee in assignees_data if assignee]
        return assignee_dfs
    elif isinstance(assignees_data, dict):

        assignee_df = assignee_to_dataframe(assignees_data)
        return [assignee_df] if not assignee_df.empty else []
    else:
        return []


def process_assignee(assignee_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['login', 'id', 'type', 'site_admin']
    existing_columns = [col for col in required_columns if col in assignee_df.columns]
    return assignee_df.loc[:, existing_columns]


def process_assignees(assignees_df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    return [process_assignee(df) for df in assignees_df_list if not df.empty]


def assignee_dataframe_to_json(assignee_df: pd.DataFrame) -> str:
    if assignee_df.empty:
        return "[]"
    assignees = assignee_df.to_dict(orient='records')
    assignee_json = json.dumps(assignees)
    return assignee_json


def assignees_dataframe_to_json(assignees_dfs: List[pd.DataFrame]) -> str:
    all_assignees = []
    for df in assignees_dfs:
        if not df.empty:
            assignees = df.to_dict(orient='records')
            all_assignees.extend(assignees)
    assignees_json = json.dumps(all_assignees)
    return assignees_json


def format_assignees(df: pd.DataFrame) -> pd.DataFrame:
    def format_assignees_column(assignees_data):
        assignees_dfs = assignees_to_dataframe_list(assignees_data)
        processed_assignees = process_assignees(assignees_dfs)
        return assignees_dataframe_to_json(processed_assignees)

    def format_assignee_column(assignee_data):
        assignee_df = assignee_to_dataframe(assignee_data)
        processed_assignee = process_assignee(assignee_df)
        return assignee_dataframe_to_json(processed_assignee)


    df['assignees'] = df['assignees'].swifter.apply(format_assignees_column)
    df['assignee'] = df['assignee'].swifter.apply(format_assignee_column)

    return df
