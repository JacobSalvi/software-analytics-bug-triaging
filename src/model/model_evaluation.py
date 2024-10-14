import argparse
from pathlib import Path
from src.model.Predictor import Predictor
from src.Database import Database
from src.utils import utils


def model_evaluation(use_gpu: bool, models_dir: Path):
    predictor = Predictor(models_dir, use_gpu)
    predictor.train_or_load()
    try:
        test_df = Database.get_test_set()
        accuracy = predictor.evaluate(test_df)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Predictor")
    argument_parser.add_argument("--models_dir",type=Path, default=utils.get_model_dir(), help="Path to the models directory")
    argument_parser.add_argument("--use_gpu", default=True, help="Use GPU for prediction")
    args = argument_parser.parse_args()
    model_evaluation(use_gpu=args.use_gpu, models_dir=args.models_dir)
