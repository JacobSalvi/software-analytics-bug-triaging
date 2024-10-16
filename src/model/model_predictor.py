import argparse
from pathlib import Path

from src.model.Predictor import Predictor

def print_candidates(candidates, issue_id: int):
    print(f"Best candidate assignees for issue {issue_id}:")
    for idx, assignee in enumerate(candidates, 1):
        print(f"{idx}. Assignee: {assignee}")

def predict_assignees(issue_id: int, models_dir: Path = None, use_gpu: bool = True, batch_size: int = 256):
    predictor = Predictor(models_dir, use_gpu, batch_size)
    predictor.load_models()
    candidates = predictor.predict_assignees(issue_id)
    return candidates


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Predictor")
    argument_parser.add_argument("--issue_id", type=int, default=752417277, help="Issue ID to predict assignees for, if not specified, a default issue ID will be used")
    argument_parser.add_argument("--models_dir",type=Path, default=None, help="Path to the models directory")
    argument_parser.add_argument("--use_gpu", default=True, help="Use GPU for prediction")
    print_candidates(predict_assignees(**vars(argument_parser.parse_args())), issue_id=argument_parser.parse_args().issue_id)
