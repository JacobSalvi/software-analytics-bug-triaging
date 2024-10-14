import argparse
from pathlib import Path
from typing import List

from src.model.Predictor import Predictor
from src.utils import utils


def print_candidates(candidates: List[str], issue_id: int):
    print(f"Best candidate assignees for issue {issue_id}:")
    for idx, assignee in enumerate(candidates, 1):
        print(f"{idx}. Assignee: {assignee}")
    return


def predict_assignees(issue_id: int, use_gpu: bool, models_dir: Path) -> List[str]:
    predictor = Predictor(models_dir, use_gpu)
    predictor.train_or_load()
    candidates = predictor.predict_assignees(issue_id)
    return candidates


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Predictor")
    argument_parser.add_argument("--issue_id", type=int, default=752417277,
                                 help="Issue ID to predict assignees for, if not specified, a default issue ID will be used")
    argument_parser.add_argument("--models_dir",type=Path, default=utils.get_model_dir(),
                                 help="Path to the models directory")
    argument_parser.add_argument("--use_gpu", default=True, help="Use GPU for prediction")
    args = argument_parser.parse_args()
    predicted_assignees = predict_assignees(issue_id=args.issue_id, use_gpu=args.use_gpu, models_dir=args.models_dir)
    print_candidates(candidates=predicted_assignees, issue_id=argument_parser.parse_args().issue_id)
