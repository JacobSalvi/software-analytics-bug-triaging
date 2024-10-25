import argparse
from pathlib import Path
from src.Database import Database
from src.model.Predictor import Predictor
from src.model.model_utils import add_default_args, get_chosen_model_dir
from src.utils import utils


def model_train(only_recent_issues: bool, use_gpu: bool = True, batch_size: int = 16, epochs: int = 5, lr: float = 2e-5):
    print(f"Training model with only recent issues: {only_recent_issues}")
    models_dir = get_chosen_model_dir(only_recent_issues)
    predictor = Predictor(models_dir, use_gpu, batch_size, epochs, lr)
    df = Database.get_train_set() if not only_recent_issues else Database.get_recent_instances()
    predictor.train(df)

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Trainer")
    argument_parser = add_default_args(argument_parser)
    args = argument_parser.parse_args()
    model_train(**vars(args))