from pathlib import Path

from src.model.Predictor import Predictor
from src.utils import utils


def model_train(use_gpu: bool = True, models_dir: Path = None):
    predictor = Predictor(models_dir, use_gpu)
    try:
        predictor.train_or_load(force_train=True)
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")

if __name__ == '__main__':
    model_train(models_dir=utils.get_model_dir())
