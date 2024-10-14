import argparse
import os
from pathlib import Path
from typing import List, Dict

import requests
from dotenv import load_dotenv
import pandas as pd

from src.utils import utils


def get_issues_raw_request(token, start_page=0) -> List[Dict]:
    payload = {}
    headers = {
        'Authorization': f'Bearer {token}'
    }
    max_issue_number: int = 220000
    parsed_issues: List[Dict] = []
    current_page = start_page
    try:
        while True:
            print(f"Current page is {current_page}")
            url = f"https://api.github.com/repos/microsoft/vscode/issues?state=closed&page={current_page}&per_page=100&direction=asc"
            response = requests.request("GET", url, headers=headers, data=payload)
            if response.status_code != 200:
                print(f"Status code {response.status_code} for page {current_page}")
                break
            issues = response.json()
            if len(issues) == 0:
                print(f"No issues for page {current_page}")
                break
            issues_to_keep = [issue for issue in issues if issue["number"] <= max_issue_number
                              and len(issue["assignees"]) == 1]

            if current_page % 100 == 0:
                print(f"Saving up to page {current_page}, number of issues {len(parsed_issues)}")
                output = utils.get_output()
                parsed_json = output.joinpath(f"parsed_issues_{current_page}.json")
                df = pd.DataFrame(parsed_issues)
                df.to_json(parsed_json, orient='records', lines=True)
            current_page += 1
            parsed_issues.extend(issues_to_keep)

            min_number = min([issue["number"] for issue in issues])
            if min_number > max_issue_number:
                print(f"Min issue number is {min_number}")
                break
    except Exception as e:
        print(f"Exception {e}")
        return parsed_issues
    return parsed_issues


def main():
    argument_parser = argparse.ArgumentParser("Perform github requests")
    argument_parser.add_argument("--starting-page", dest="start_page",type=int, default=0)
    args = argument_parser.parse_args()
    starting_page = args.start_page
    output = utils.get_output()
    if not output.is_dir():
        output.mkdir()
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")

    parsed_csv: Path = output.joinpath("parsed_issues.csv")
    parsed_issues = get_issues_raw_request(github_token, starting_page)

    print(f"Saving parsed issues to {parsed_csv}")
    df = pd.DataFrame(parsed_issues)
    df.to_csv(parsed_csv, index=False)

    parsed_json: Path = output.joinpath("parsed_issues.json")
    df.to_json(parsed_json, orient='records', lines=True)
    

if __name__ == "__main__":
    main()
