from email.parser import Parser

from src.utils import utils


def add_default_args(parser):
    parser.add_argument("--only_recent_issues", default=True, help="Evaluate only on early instances")
    parser.add_argument("--use_gpu", default=True, help="Use GPU for prediction")
    parser.add_argument("--batch_size", default=16, help="Define the batch size for training")
    parser.add_argument("--epochs", default=5, help="Define the maximum number of iterations for training")
    parser.add_argument("--lr", default=2e-5, help="Define the learning rate for training")
    return parser

def get_chosen_model_dir(only_recent_issues: bool):
    return utils.get_models_recenet_dir() if only_recent_issues else utils.get_model_dir()