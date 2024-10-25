#ORRECT
import json
import os
import tarfile
from pathlib import Path
import io

import pandas as pd
from pandas import DataFrame
from sympy import false

from src.processing.issues_processing import process_data
from src.utils import utils
from src.utils.utils import compress_file_to_tar_gz


class DataHandler:
    PARSED_TAR_GZ_FILE_PATH = utils.data_dir().joinpath("cleaned_parsed_issues.tar.gz")
    TAR_GZ_FILE_1 = utils.data_dir().joinpath("raw_parsed_issues_1.tar.gz")
    TAR_GZ_FILE_2 = utils.data_dir().joinpath("raw_parsed_issues_2.tar.gz")

    COMMITS = utils.data_dir().joinpath("commits_per_user.csv")

    @staticmethod
    def get_commits() -> DataFrame:
        return pd.read_csv(DataHandler.COMMITS, header=None)

    @staticmethod
    def get_raw() -> list:
        combined_data = []

        if not Path(DataHandler.TAR_GZ_FILE_1).exists():
            raise FileNotFoundError(f"The file {DataHandler.TAR_GZ_FILE_1.name} does not exist, "
                                    f"please execute the scraper first issue_scraper.py and or data_splitter.py")

        if not Path(DataHandler.TAR_GZ_FILE_2).exists():
            raise FileNotFoundError(f"The file {DataHandler.TAR_GZ_FILE_2.name} does not exist, "
                                    f"please execute the scraper first issue_scraper.py and or data_splitter.py")

        combined_data.extend(
            DataHandler.extract_json_from_tar(DataHandler.TAR_GZ_FILE_1))
        combined_data.extend(
            DataHandler.extract_json_from_tar(DataHandler.TAR_GZ_FILE_2))

        return combined_data

    @staticmethod
    def extract_json_from_tar(tar_gz_file: Path) -> list:
        json_data = []

        if not Path(tar_gz_file).exists():
            raise FileNotFoundError(f"The file {tar_gz_file} does not exist")

        with tarfile.open(tar_gz_file, "r:gz") as tar:

            for member in tar.getmembers():
                if member.name.endswith(".json"):
                    with tar.extractfile(member) as f:
                        if f:
                            for line in f:
                                json_data.append(json.loads(line.decode('utf-8')))
                    break

        if not json_data:
            raise FileNotFoundError(f"No JSON file was found in {tar_gz_file}")

        return json_data

    @staticmethod
    def get_parsed(force_parse: bool = false) -> DataFrame:
        if force_parse and Path(DataHandler.PARSED_TAR_GZ_FILE_PATH).exists():
            os.remove(DataHandler.PARSED_TAR_GZ_FILE_PATH)

        if not Path(DataHandler.PARSED_TAR_GZ_FILE_PATH).exists():
            df = process_data(pd.DataFrame(DataHandler.get_raw()))

            # Convert DataFrame to CSV in memory using BytesIO
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            compress_file_to_tar_gz(
                DataHandler.PARSED_TAR_GZ_FILE_PATH,
                csv_buffer,
                'cleaned_parsed_issues.csv'
            )

        with tarfile.open(DataHandler.PARSED_TAR_GZ_FILE_PATH, "r:gz") as tar:

            for member in tar.getmembers():
                if member.name.endswith(".csv"):

                    with tar.extractfile(member) as f:
                        csv_data = pd.read_csv(f)
                    return csv_data

        raise FileNotFoundError("No CSV file found in the tar.gz archive.")


if __name__ == '__main__':
    try:
        DataHandler.get_parsed(True)
    except FileNotFoundError as e:
        print(e)
