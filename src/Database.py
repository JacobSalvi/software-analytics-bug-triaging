import ast
from typing import List, Optional

import pandas as pd

from src.DataHandler import DataHandler


class Database:
    RAW_DATA = None
    DATA = None
    COMMITS = None

    @staticmethod
    def get_commits() -> pd.DataFrame:
        if Database.COMMITS is None:
            Database.COMMITS = DataHandler.get_commits()
        return Database.COMMITS

    @staticmethod
    def get_commits_per_user(user: str)  -> int:
        commits = Database.get_commits()
        return commits[commits[0] == user][1].values[0]

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
    def get_issues_by_number(number: int) -> pd.DataFrame:
        df = Database.get_issues()
        filtered_df = df[df['number'] == number]
        if filtered_df.empty:
            raise ValueError(f"No issue found with number: {number}")
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

        test_set_df.loc[:, 'labels'] = test_set_df['labels'].apply(Database.combine_labels)
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

        train_df.loc[:, 'labels'] = train_df['labels'].apply(Database.combine_labels)
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

    @staticmethod
    def get_all_assignees_in_issues()-> List[str]:
        df = Database.get_issues()
        return df['assignee'].unique()

    @staticmethod
    def get_user_by_id(assignee_id: int) -> Optional[str]:
        assignees = Database.get_all_assignees_in_issues()
        assignees = [ast.literal_eval(assignee) for assignee in assignees]
        return next((assignee for assignee in assignees if assignee.get('id') == assignee_id), None)


    @staticmethod
    def combine_labels(labels):
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)

        if labels:
            combined_string = ' '.join(labels).upper()
        else:
            combined_string = ''

        return combined_string



if __name__ == '__main__':

    df = Database.get_test_set()
    pass




