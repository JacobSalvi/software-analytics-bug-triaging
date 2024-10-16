import argparse
from pathlib import Path

from src.Database import Database
from src.model.Predictor import Predictor

def print_candidates(candidates: list[str], issue_id: int):
    print(f"Best candidate assignees for issue {issue_id}:")
    for idx, assignee_id in enumerate(candidates, 1):
        user = Database.get_user_by_id(assignee_id)
        print(f"{idx}. Assignee: {assignee_id}")

def get_users_from_candidates(candidates) -> list[str]:
    return [Database.get_user_by_id(candidate) for candidate in candidates]

def get_commits_by_user_name(user_names: list[str]) -> list[int]:
    return [Database.get_commits_per_user(user_name) for user_name in user_names]

def get_user_names_from_user(users: list[str]) -> list[str]:
    return [user.get("login") for user in users]


def predict_assignees(issue_id: int, models_dir: Path = None, use_gpu: bool = True, batch_size: int = 256):
    predictor = Predictor(models_dir, use_gpu, batch_size)
    predictor.load_models()
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
