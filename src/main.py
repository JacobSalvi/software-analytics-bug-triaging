import os
from dotenv import load_dotenv
from github import Auth, Github


def main():
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    auth = Auth.Token(github_token)
    github = Github(auth=auth)
    repo = github.get_repo("microsoft/vscode")

    issues = repo.get_issues(state='closed')

    for issue in issues:
        print(issue)
        if issue.number < 100:
            print(f"Issue ID: {issue.number}, Title: {issue.title}, State: {issue.state}")
    

if __name__ == "__main__":
    main()
