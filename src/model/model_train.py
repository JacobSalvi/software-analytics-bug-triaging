from pathlib import Path

from src.model.Predictor import Predictor

def model_train(use_gpu: bool = True, models_dir: Path = None):
    predictor = Predictor(models_dir, use_gpu)
    predictor.train_or_load(force_train=True)

if __name__ == '__main__':
    model_train()
