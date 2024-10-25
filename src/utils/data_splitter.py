## CORRECT
import json
from pathlib import Path
from src.utils import utils
from src.utils.utils import compress_file_to_tar_gz

def split_ndjson_file(input_file: Path, output_file1: Path, output_file2: Path, verbose: bool = False):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    mid_index = len(data) // 2
    first_half = data[:mid_index]
    second_half = data[mid_index:]

    with open(output_file1, 'w') as f1:
        for item in first_half:
            f1.write(json.dumps(item) + '\n')

    with open(output_file2, 'w') as f2:
        for item in second_half:
            f2.write(json.dumps(item) + '\n')

    if verbose:
        print(f"Data has been split into {output_file1.name} and {output_file2.name}")

    try:
        check_file_readable(output_file1)
        check_file_readable(output_file2)
    except Exception as e:
        print(f"Error while verifying files: {e}")
        return

    if verbose:
        print(f"Both {output_file1.name} and {output_file2.name} are valid JSON files.")


def check_file_readable(file_path: Path):
    with open(file_path, 'r') as f:
        for line in f:
            json.loads(line)

def divide_json_file(path: Path):
    if not path.exists():
        print(f"File {path.name} does not exist, please execute the scraper first issue_scraper.py")
        return

    csv1_path = path.parent / "raw_parsed_issues_1.json"
    csv2_path = path.parent / "raw_parsed_issues_2.json"
    split_ndjson_file(path, csv1_path, csv2_path)

    try:
        tar_gz1_path = path.parent / "raw_parsed_issues_1.tar.gz"
        with open(csv1_path, 'rb') as file1:
            compress_file_to_tar_gz(tar_gz1_path, file1, "raw_parsed_issues_1.json")

        tar_gz2_path = path.parent / "raw_parsed_issues_2.tar.gz"
        with open(csv2_path, 'rb') as file2:
            compress_file_to_tar_gz(tar_gz2_path, file2, "raw_parsed_issues_2.json")

        csv1_path.unlink()
        csv2_path.unlink()
        path.unlink()
        print(f"Created {tar_gz1_path.name} and {tar_gz2_path.name}. Deleted intermediate CSV files.")

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    divide_json_file(utils.data_dir().joinpath("raw_parsed_issues.json"))

if __name__ == "__main__":
    main()
