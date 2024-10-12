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
    def get_all_issues_with_one_assignee() -> pd.DataFrame:
        df = Database.get_issues()
        filtered_df = df[df['assignees'] == 1]

        if filtered_df.empty:
            raise ValueError("No issues found with one assignee")

        return filtered_df