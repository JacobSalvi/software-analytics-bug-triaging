import json
import tarfile
from pathlib import Path
import io

import pandas as pd
from pandas import DataFrame

from src.processing.issues_processing import process_data


class DataHandler:
    PARSED_TAR_GZ_FILE_PATH = '../data/cleaned_parsed_issues.tar.gz'

    TAR_GZ_FILE_1 = '../data/raw_parsed_issues_1.tar.gz'
    TAR_GZ_FILE_2 = '../data/raw_parsed_issues_2.tar.gz'


    @staticmethod
    def get_raw() -> list:
        """
        Reads the JSON content from two specified JSON files compressed in two separate tar.gz archives
        and returns the combined list of JSON objects.

        Returns:
        - A list containing the combined data from both files.
        """
        combined_data = []

        combined_data.extend(
            DataHandler.extract_json_from_tar(DataHandler.TAR_GZ_FILE_1))
        combined_data.extend(
            DataHandler.extract_json_from_tar(DataHandler.TAR_GZ_FILE_2))

        return combined_data

    @staticmethod
    def extract_json_from_tar(tar_gz_file: str) -> list:
        """
        Extracts the first JSON file (NDJSON format) from a specified tar.gz archive and returns the content as a list of JSON objects.

        Parameters:
        - tar_gz_file: The path to the tar.gz archive.

        Returns:
        - A list of JSON objects from the first JSON file found in the tar.gz.
        """
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
    def get_parsed() -> DataFrame:
        """Extracts and reads the single CSV file from a tar.gz archive and returns it as a DataFrame"""

        if not Path(DataHandler.PARSED_TAR_GZ_FILE_PATH).exists():
            df = process_data(pd.DataFrame(DataHandler.get_raw()))

            # Convert DataFrame to CSV in memory using BytesIO
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)  # Reset buffer position to the beginning

            DataHandler.compress_file_to_tar_gz(
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


    @staticmethod
    def compress_file_to_tar_gz(output_filename, file_obj, file_name):
        """
        Compresses a single file-like object into a tar.gz archive.
        """
        with tarfile.open(output_filename, "w:gz") as tar:
            tarinfo = tarfile.TarInfo(name=file_name)
            file_obj.seek(0, io.SEEK_END)  # Get the size of the file
            tarinfo.size = file_obj.tell()
            file_obj.seek(0)  # Reset buffer position to the beginning
            tar.addfile(tarinfo, file_obj)

        return output_filename


if __name__ == '__main__':
    DataHandler.get_parsed()
