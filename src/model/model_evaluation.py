import argparse
from src.model.Predictor import Predictor
from src.Database import Database
from src.model.model_utils import add_default_args, get_chosen_model_dir


def model_evaluation(evaluate_only_early, use_gpu: bool = True, batch_size: int = 16,  epochs: int = 5, lr: float = 2e-5) -> float:
    models_dir = get_chosen_model_dir(evaluate_only_early)
    predictor = Predictor(models_dir, use_gpu, batch_size, epochs, lr)
    predictor.load_models()
    try:
        test_df = Database.get_test_set()
        accuracy = predictor.evaluate(test_df)
        return accuracy
    except Exception as e:
        print(f"Evaluation failed: {e}")

def print_statistics(accuracy: float):
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Evaluation")
    argument_parser = add_default_args(argument_parser)
    args = argument_parser.parse_args()
    evaluation = model_evaluation(evaluate_only_early=args.only_recent_issues,
                                  use_gpu=args.use_gpu,
                                  batch_size=args.batch_size)
    print_statistics(evaluation)

