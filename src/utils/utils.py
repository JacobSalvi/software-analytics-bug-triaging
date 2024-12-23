import io
import shutil
import tarfile
from pathlib import Path


def get_model_dir() -> Path:
    model_path = Path(__file__).parents[2].joinpath("models")
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path

def data_dir() -> Path:
    data_path = Path(__file__).parents[2].joinpath("data")
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path

def get_models_recent_dir() -> Path:
    models_recent_path = Path(__file__).parents[2].joinpath('models_recent')
    models_recent_path.mkdir(parents=True, exist_ok=True)
    return models_recent_path

def remove_all_files_and_subdirectories_in_folder(folder_path: Path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"The folder {folder_path} does not exist or is not a directory.")

    for item in folder.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()

def compress_file_to_tar_gz(output_filename, file_obj, file_name):
    with tarfile.open(output_filename, "w:gz") as tar:
        tarinfo = tarfile.TarInfo(name=file_name)
        file_obj.seek(0, io.SEEK_END)
        tarinfo.size = file_obj.tell()
        file_obj.seek(0)
        tar.addfile(tarinfo, file_obj)

    return output_filename