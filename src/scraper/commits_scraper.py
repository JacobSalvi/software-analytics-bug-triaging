import pandas as pd
import requests
from dotenv import load_dotenv
import os
import pprint
from pathlib import Path


assignee = "Yoyokrazy"

def get_output() -> Path:
    return Path(__file__).parents[1].joinpath('output')

# TODO: Get the number of commits for all branches, not just main one

def fetch_commits_for_assignees(assignee, token, users):
    #gets the number of commits in main branch
    
    commits_per_user = {}
    for user in users:
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
        commits_per_user[user] = number_of_commits
    return commits_per_user


def get_assigness():
    pass

def main():
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    assignees = get_assigness()
    commits_per_user = fetch_commits_for_assignees(assignee, github_token, assignees)
    df = pd.DataFrame(commits_per_user)
    output = get_output()
    path = output.joinpath("commits_per_user.csv")
    df.to_csv(path, index=False)
   
if __name__ == '__main__':
    main()

  