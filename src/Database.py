import ast
from typing import List

import pandas as pd

from src.DataHandler import DataHandler


class Database:
    RAW_DATA = None
    DATA = None

    @staticmethod
    def get_raw_issues() -> pd.DataFrame:
        if Database.RAW_DATA is None:
            Database.RAW_DATA = DataHandler.get_raw()
        return Database.RAW_DATA

    # all issues have only one assignee
    @staticmethod
    def get_issues() -> pd.DataFrame:
        if Database.DATA is None:
            Database.DATA = DataHandler.get_parsed()
        return Database.DATA

    @staticmethod
    def get_issues_by_id(issue_id: int) -> pd.DataFrame:
        df = Database.get_issues()
        filtered_df = df[df['id'] == issue_id]

        if filtered_df.empty:
            raise ValueError(f"No issue found with id: {issue_id}")

        return filtered_df

    @staticmethod
    def get_test_set() -> pd.DataFrame:
        df = Database.get_issues()
        test_set_df = df[
            (df['number'] >= 210001) &
            (df['number'] <= 220000)
        ]
        # Every assignee on the test set should appear at least once in the training set
        train_assignees = set(Database.get_train_set()['assignee'].unique())
        test_set_df = test_set_df[test_set_df['assignee'].isin(train_assignees)]

        if test_set_df.empty:
            raise ValueError("Empty test set")

        return test_set_df

    @staticmethod
    def get_train_set() -> pd.DataFrame:
        df = Database.get_issues()
        train_set_df = df[(df['number'] <= 210000)]
        if train_set_df.empty:
            raise ValueError("Empty train set")

        # remove developers who have been assignees only a few times- -> model work better with fewer data
        assignee_counts = train_set_df['assignee'].value_counts()
        assignees_to_keep = assignee_counts[assignee_counts > 5].index
        train_df = train_set_df[train_set_df['assignee'].isin(assignees_to_keep)]
        train_df.dropna(subset=['title', 'body', 'assignee'], inplace=True)
        return train_df

    @staticmethod
    def get_recent_instances() -> pd.DataFrame:
        df = Database.get_issues()
        recent_instances_df = df[
            (df['number'] >= 190000) &
            (df['number'] <= 210000)
        ]
        if recent_instances_df.empty:
            raise ValueError("Empty recent instances set")
        return recent_instances_df

    @staticmethod
    def extract_assignee_ids(df: pd.DataFrame) -> List[int]:
        return [
            assignee.get('id') if isinstance(assignee := ast.literal_eval(assignee_str), dict) else None
            for assignee_str in df['assignee']
        ]


if __name__ == '__main__':
    pass




