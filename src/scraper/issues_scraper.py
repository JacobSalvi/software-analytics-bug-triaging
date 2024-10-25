## CORRECT
import argparse
import os
from pathlib import Path
from typing import List, Dict

import requests
from dotenv import load_dotenv
import pandas as pd

from src.utils import utils


def get_issues_raw_request(token: str, start_page: int = 0, max_issue_number: int = 220000, save_intermediate_json: bool = False) -> List[Dict]:
    payload = {}
    headers = {
        'Authorization': f'Bearer {token}'
    }
    c_max_issue_number: int = max_issue_number
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
            issues_to_keep = [issue for issue in issues if issue["number"] <= c_max_issue_number
                              and len(issue["assignees"]) == 1]

            if current_page % 100 == 0:
                if save_intermediate_json:
                    print(f"Saving up to page {current_page}, number of issues {len(parsed_issues)}")
                    output = utils.data_dir()
                    parsed_json = output.joinpath(f"parsed_issues_{current_page}.json")
                    df = pd.DataFrame(parsed_issues)
                    df.to_json(parsed_json, orient='records', lines=True)
            current_page += 1
            parsed_issues.extend(issues_to_keep)

            min_number = min([issue["number"] for issue in issues])
            if min_number > c_max_issue_number:
                print(f"Min issue number is {min_number}")
                break
    except Exception as e:
        print(f"Exception {e}")
        return parsed_issues
    return parsed_issues


def main():
    argument_parser = argparse.ArgumentParser("Perform github VS-Code issues scraping")
    argument_parser.add_argument("--starting_page", dest="start_page",type=int, default=0, help="Starting page for scraping")
    argument_parser.add_argument("--max_issue_number", type=int, default=2000, help="Max issue number to scrape")
    args = argument_parser.parse_args()
    output = utils.data_dir()
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")

    parsed_issues = get_issues_raw_request(github_token, args.start_page, args.max_issue_number, False)

    raw_json: Path = output.joinpath("raw_parsed_issues.json")
    print(f"Saving parsed issues to {raw_json}")
    df = pd.DataFrame(parsed_issues)
    df.to_json(raw_json, orient='records', lines=True)


if __name__ == "__main__":
    main()
