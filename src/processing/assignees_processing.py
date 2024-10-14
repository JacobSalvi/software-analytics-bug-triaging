import pandas as pd
import swifter

required_columns = ['login', 'id', 'type', 'site_admin']

def filter_assignees(assignees):
    if assignees is None:
        return []
    return [
        {key: value for key, value in assignee.items() if key in required_columns}
        for assignee in assignees if assignee is not None
    ]

def filter_assignee(assignee):
    if assignee is None:
        return {}
    return {key: value for key, value in assignee.items() if key in required_columns}

def filter_assignee_data(df: pd.DataFrame) -> pd.DataFrame:
    # Apply swifter to process data with parallelism
    #df['assignees'] = df['assignees'].swifter.apply(filter_assignees)
    df['assignee'] = df['assignee'].swifter.apply(filter_assignee)
    return df
