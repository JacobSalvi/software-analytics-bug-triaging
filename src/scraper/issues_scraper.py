import argparse
import os
from pathlib import Path
from typing import List, Dict
from urllib.request import DataHandler

import requests
from dotenv import load_dotenv
import pandas as pd

from src.utils import utils
from src.utils.data_splitter import split_ndjson_file


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


def divide_json_file(path: Path):
    # Define paths for the split CSV files
    csv1_path = path.parent / "raw_parsed_issues_1.csv"
    csv2_path = path.parent / "raw_parsed_issues_2.csv"
    split_ndjson_file(path, csv1_path, csv2_path)

    try:
        tar_gz1_path = path.parent / "raw_parsed_issues_1.tar.gz"
        with open(csv1_path, 'rb') as file1:
            DataHandler.compress_file_to_tar_gz(tar_gz1_path, file1, "raw_parsed_issues_1.csv")

        tar_gz2_path = path.parent / "raw_parsed_issues_2.tar.gz"
        with open(csv2_path, 'rb') as file2:
            DataHandler.compress_file_to_tar_gz(tar_gz2_path, file2, "raw_parsed_issues_2.csv")

        csv1_path.unlink()
        csv2_path.unlink()
        path.unlink()
        print(f"Successfully created {tar_gz1_path} and {tar_gz2_path}. Deleted intermediate CSV files.")

    except Exception as e:
        print(f"An error occurred during compression or deletion: {e}")


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
