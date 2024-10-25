import argparse

import pandas as pd
import requests
from dotenv import load_dotenv
import os
from src.Database import Database
import ast
from typing import List, Dict, AnyStr

from src.utils import utils


def fetch_commits_for_assignees(token: str, assignees: List[str]) -> Dict[str, int]:
    #gets the number of commits in main branch
    commits_per_user = {}
    for assignee in assignees:
        url = f"https://api.github.com/repos/microsoft/vscode/commits?author={assignee}"
        headers = {
        'Authorization': f'Bearer {token}'
        }
        number_of_commits = 0
        while url:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                commits = response.json()
                number_of_commits += len(commits)
                if 'Link' in response.headers:
                    links = response.headers['Link']
                    next_url = None
                    for link in links.split(','):
                        if 'rel="next"' in link:
                            next_url = link[link.index('<') + 1:link.index('>')]
                    url = next_url  
                else:
                    url = None 
            else:
                print(f"Error: {response.status_code} - {response.text}")
                break
        commits_per_user[assignee] = number_of_commits
    return commits_per_user


def get_assignees(df: pd.DataFrame, appearance_threshold: int = 5) -> List[str]:
    return df['assignee'].value_counts()[lambda x: x >= appearance_threshold].index.tolist()


def get_assignees_logins(assignees: List[AnyStr]) -> List[str]:
    logins = []
    for assignee in assignees:
        try:
            if 'login' in (assignee_dict:= ast.literal_eval(assignee)):
                logins.append(assignee_dict['login'])
        except Exception as e:
            print("Exception: {e}")
            print(assignee)        
    return logins


def main():
    df = Database.get_issues()
    logins = get_assignees_logins(get_assignees(df))
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    commits_per_user = fetch_commits_for_assignees(github_token, logins)
    series = pd.Series(commits_per_user)
    output = utils.data_dir()
    path = output.joinpath("commits_per_user.csv")
    series.to_csv(path, header=False)

if __name__ == '__main__':
    main()

  