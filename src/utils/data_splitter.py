import json
from pathlib import Path

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


if __name__ == "__main__":
    # Example usage
    input_file = Path('../../data/parsed_issues.json')
    output_file1 = Path('../../data/parsed_issues_1.json')
    output_file2 = Path('../../data/parsed_issues_2.json')

    split_ndjson_file(input_file, output_file1, output_file2)
