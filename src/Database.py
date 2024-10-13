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
            (df['id'] >= 210001) &
            (df['id'] <= 220000)]
        if test_set_df.empty:
            raise ValueError("No issues found in the test set with the given criteria")
        return test_set_df

    @staticmethod
    def get_train_set() -> pd.DataFrame:
        df = Database.get_issues()
        train_set__df = df[~((df['id'] >= 210001) & (df['id'] <= 220000))]
        if train_set__df.empty:
            raise ValueError("No issues found in the test set with the given criteria.")
        return train_set__df

    @staticmethod
    def get_recent_instances() -> pd.DataFrame:
        df = Database.get_issues()
        recent_instances_df = df[
            (df['id'] >= 190000) &
            (df['id'] <= 210000)
        ]

        if recent_instances_df.empty:
            raise ValueError("No recent instances found with the given criteria.")

        return recent_instances_df


if __name__ == '__main__':
    print(Database.get_issues_by_id(752417277))
    print(f"Number of rows: {Database.get_test_set().shape[0]}")