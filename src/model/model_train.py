import argparse
from pathlib import Path

from src.model.Predictor import Predictor

def model_train(models_dir: Path = None, use_gpu: bool = True, batch_size: int = 256):
    predictor = Predictor(models_dir, use_gpu, batch_size)
    predictor.train()

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Trainer")
    argument_parser.add_argument("--models_dir",type=Path, default=None, help="Path to the models directory")
    argument_parser.add_argument("--use_gpu", default=True, help="Use GPU for prediction")
    argument_parser.add_argument("--batch_size", default=True, help="Define the batch size for training")
    args = argument_parser.parse_args()
    model_train(**vars(args))