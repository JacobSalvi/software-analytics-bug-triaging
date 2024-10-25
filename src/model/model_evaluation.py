import argparse
from typing import Dict, Any

from src.model.Predictor import Predictor
from src.Database import Database
from src.model.model_utils import add_default_args, get_chosen_model_dir, get_chosen_train_set


def model_evaluation(only_recent_issues, use_gpu: bool = True, batch_size: int = 16,  epochs: int = 5, lr: float = 2e-5) -> Dict[str, Any]:
    print(f"Evaluating model with only recent issues: {only_recent_issues}")
    models_dir = get_chosen_model_dir(only_recent_issues)
    predictor = Predictor(models_dir, use_gpu, batch_size, epochs, lr)
    predictor.load_models()
    try:

        test_df = Database.get_test_set(get_chosen_train_set(only_recent_issues))
        accuracy = predictor.evaluate(test_df)
        return accuracy
    except Exception as e:
        print(f"Evaluation failed: {e}")


def print_statistics(results):
    accuracy = results.get("accuracy", 0) * 100
    print(f"{'Test Accuracy:':<25} {accuracy:.2f}%\n")    
    classification_report = results.get("classification_report", {})
    if not classification_report:
        print("No classification report available.\n")
        return    
    assignee_stats = []
    for assignee, metrics in classification_report.items():
        if assignee in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
            continue
        try:
            username = Database.get_user_by_id(assignee)
            assignee_name = username['login'] if 'login' in username else f"ID {assignee}"
        except Exception as e:
            assignee_name = f"ID {assignee} (Info Unavailable)"
            print(f"Warning: Could not retrieve user for assignee ID {assignee}: {str(e)}")
        
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        predictions_per_class = results.get('predictions_per_class', {}).get(assignee, 0)
        correct_predictions = results.get('correct_predictions_per_class', {}).get(assignee, 0)
        assignee_stats.append({
            'Assignee': assignee_name,
            'Precision': precision,
            'Recall': recall,
            'Total Predictions': predictions_per_class,
            'Correct Predictions': correct_predictions
        })
    assignee_stats_sorted = sorted(assignee_stats, key=lambda x: x['Precision'], reverse=True)    
    print(f"{'Assignee':<20} {'Precision':<10} {'Recall':<10} {'Total Predictions':<20} {'Correct Predictions':<20}")
    print("-" * 80)
    
    for stats in assignee_stats_sorted:
        print(f"{stats['Assignee']:<20} {stats['Precision']:<10.3f} {stats['Recall']:<10.3f} "
              f"{stats['Total Predictions']:<20} {stats['Correct Predictions']:<20}")
    print("\n")

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser("Evaluation")
    argument_parser = add_default_args(argument_parser)
    args = argument_parser.parse_args()
    evaluation = model_evaluation(**vars(args))
    print_statistics(evaluation)

