import os
from typing import List, Dict

from dotenv import load_dotenv
from github import Auth, Github
from github.PaginatedList import PaginatedList


def parse_issues(issues: PaginatedList):
    max_issue_number: int = 210000
    parsed_issues: List[Dict] = []
    for issue in issues:
        if issue.number > max_issue_number:
            continue
        if len(issue.assignees) > 1:
            continue
        parsed_issues.append(issue.raw_data)
        pass
    return


def main():
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    auth = Auth.Token(github_token)
    github = Github(auth=auth)
    repo = github.get_repo("microsoft/vscode")

    issues = repo.get_issues(state='closed')

    parse_issues(issues)
    

if __name__ == "__main__":
    main()
