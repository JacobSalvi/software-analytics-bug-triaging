import shutil
from pathlib import Path


def get_model_dir() -> Path:
    return Path(__file__).parents[2].joinpath("models")


def data_dir() -> Path:
    return Path(__file__).parents[2].joinpath("data")


def get_output() -> Path:
    return Path(__file__).parents[2].joinpath('output')


def remove_all_files_and_subdirectories_in_folder(folder_path: Path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"The folder {folder_path} does not exist or is not a directory.")

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()