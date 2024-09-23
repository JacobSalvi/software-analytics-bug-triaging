import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from github import Auth, Github
from github.PaginatedList import PaginatedList
import pandas as pd


def parse_issues(issues: PaginatedList):
    max_issue_number: int = 210000
    parsed_issues: List[Dict] = []
    current_page = 0
    while True:
        page_issues = issues.get_page(current_page)
        if len(page_issues) == 0:
            break
        # issues_to_keep = [issue.raw_data for issue in page_issues
        #                   if issue.number <=max_issue_number
        #                   and len(issue.assignees) == 1]
        issues_to_keep = [i.raw_data for i in page_issues]
        current_page += 1
        parsed_issues.extend(issues_to_keep)
        if len(parsed_issues) > 0:
            break
    return parsed_issues


def get_output() -> Path:
    return Path(__file__).parents[1].joinpath('output')


def main():
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    auth = Auth.Token(github_token)
    github = Github(auth=auth, per_page=1000)
    repo = github.get_repo("microsoft/vscode")

    issues = repo.get_issues(state='closed', sort="asc")
    output = get_output()
    if not output.is_dir():
        output.mkdir()
    parsed_csv = output.joinpath("parsed_issues.csv")
    parsed_issues = parse_issues(issues)
    df = pd.DataFrame(parsed_issues)
    df.to_csv(parsed_csv, index=False)
    

if __name__ == "__main__":
    main()
