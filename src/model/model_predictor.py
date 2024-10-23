import argparse
from pathlib import Path

from src.Database import Database
from src.model.Predictor import Predictor
from src.model.model_utils import add_default_args, get_chosen_model_dir
from src.utils import utils


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


def predict_assignees(issue_id: int, top_n: int, only_recent_issues: bool, use_gpu: bool = True, batch_size: int = 16):
    models_dir = get_chosen_model_dir(only_recent_issues)
    predictor = Predictor(models_dir, use_gpu, batch_size)
    candidates = predictor.predict_assignees(issue_id, top_n)
    return candidates


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Predictor")
    argument_parser.add_argument("--issue_id", type=int, default=752417277,
                                 help="Issue ID to predict assignees for, if not specified, a default issue ID will be used")
    argument_parser.add_argument("--top_n", default=5, help="Number of top candidates to display")
    argument_parser = add_default_args(argument_parser)
    args = argument_parser.parse_args()
    predicted_assignees = predict_assignees(args.issue_id, args.top_m, **vars(args))
    print_candidates(candidates=predicted_assignees, issue_id=argument_parser.parse_args().issue_id)
