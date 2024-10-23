import argparse
from src.model.Predictor import Predictor
from src.Database import Database
from src.model.model_utils import add_default_args, get_chosen_model_dir


def model_evaluation(evaluate_only_early, use_gpu: bool = True, batch_size: int = 256):
    models_dir = get_chosen_model_dir(evaluate_only_early)
    predictor = Predictor(models_dir, use_gpu, batch_size)
    predictor.load_models()
    try:
        test_df = Database.get_test_set()
        accuracy = predictor.evaluate(test_df)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Predictor")
    argument_parser = add_default_args(argument_parser)
    args = argument_parser.parse_args()
    model_evaluation(**vars(args))
