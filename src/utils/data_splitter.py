import argparse
import json
from pathlib import Path

#from src.DataHandler import DataHandler


#from src.DataHandler import DataHandler


def split_ndjson_file(input_file: Path, output_file1: Path, output_file2: Path):
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

    print(f"Data has been split into {output_file1} and {output_file2}")

    try:
        check_file_readable(output_file1)
        check_file_readable(output_file2)
    except Exception as e:
        print(f"Error while verifying files: {e}")
        return

    print(f"Both {output_file1} and {output_file2} are valid JSON files.")


def check_file_readable(file_path: Path):
    with open(file_path, 'r') as f:
        for line in f:
            json.loads(line)

"""
def divide_json_file(path: Path):
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
"""


def main():
    argument_parser = argparse.ArgumentParser("Data Splitter")
    argument_parser.add_argument("json_data", type=str, help="the raw JSON file to split and compress")
    #divide_json_file(Path(argument_parser.parse_args().json_data))

if __name__ == "__main__":
    main()
