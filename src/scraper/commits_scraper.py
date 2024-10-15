import pandas as pd
import requests
from dotenv import load_dotenv
import os
import pprint
from pathlib import Path
from src.DataHandler import DataHandler
from src.Database import Database
import ast
from github import Github
from typing import List, Dict, Any

def get_output() -> Path:
    return Path(__file__).parents[1].joinpath('output')

# TODO: Get the number of commits for all branches, not just main one

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
            print(number_of_commits)
        commits_per_user[assignee] = number_of_commits
    return commits_per_user


def get_assignees(df: pd.DataFrame, appearance_threshold: int = 5) -> List[str]:
    return df['assignee'].value_counts()[lambda x: x >= appearance_threshold].index.tolist()


def get_assignees_logins(assignees: pd.Series) -> List[str]:
    logins = []
    for assignee in assignees:
        try:
            if 'login' in (assignee_dict:= ast.literal_eval(assignee)):
                logins.append(assignee_dict['login'])
        except Exception as e:
            print("Exception: {e}")
            print(assignee)        
    return logins

# def get_commit_count(username, token):
#     g = Github(token)
#     repo = g.get_repo(f"microsoft/vscode")
#     commit_count = 0
#     for commit in repo.get_commits(author=username):
#         commit_count += 1
#         print(commit_count)

#     return commit_count

def main():
    df = Database.get_issues()
    logins = get_assignees_logins(get_assignees(df))
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    commits_per_user = fetch_commits_for_assignees(github_token, logins)
    df = pd.DataFrame(commits_per_user)
    output = get_output()
    path = output.joinpath("commits_per_user.csv")
    df.to_csv(path, index=False)
   
if __name__ == '__main__':
    main()

  