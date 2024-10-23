import argparse
from pathlib import Path
from src.Database import Database
from src.model.Predictor import Predictor
from src.model.model_utils import add_default_args, get_chosen_model_dir
from src.utils import utils


def model_train(evaluate_only_early, use_gpu: bool = True, batch_size: int = 256):
    models_dir = get_chosen_model_dir(evaluate_only_early)
    predictor = Predictor(models_dir, use_gpu, batch_size)
    df = Database.get_train_set() if not evaluate_only_early else Database.get_recent_instances()
    predictor.train(df)

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Trainer")
    argument_parser.add_argument("--only_recent_issues", default=False, help="Evaluate only on early instances")
    argument_parser = add_default_args(argument_parser)
    args = argument_parser.parse_args()
    model_train(**vars(args))