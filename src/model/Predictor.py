from pathlib import Path
from typing import List, Callable
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
from tqdm import tqdm

from src.Database import Database
from src.utils import utils
from src.utils.utils import remove_all_files_and_subdirectories_in_folder, get_model_dir, get_models_recenet_dir

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5


def get_corpus(train_df: pd.DataFrame) -> List[str]:
    return (train_df['title'] + ' ' + train_df['body'] + ' ' + train_df['labels']).tolist()


def preprocess_text(text: str) -> str:
    return text.lower().strip()


def get_assignee_ids(train_df: pd.DataFrame) -> List[int]:
    return Database.extract_assignee_ids(train_df)


def get_query_corpus(row: pd.DataFrame) -> str:
    row = row.fillna('')
    query_corpus = row.iloc[0]['title'] + ' ' + row.iloc[0]['body'] + ' ' + row.iloc[0]['labels']
    return preprocess_text(query_corpus)


class Predictor:
    def __init__(self, model_dir: Path = utils.get_model_dir(), use_gpu: bool = True,
                 batch_size: int = BATCH_SIZE, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE):
        self.model = None
        self.tokenizer = None
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.BATCH_SIZE = batch_size

        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.label_encoder_path = self.model_dir.joinpath('label_encoder.joblib')
        self.tokenizer_path = self.model_dir.joinpath('tokenizer')
        self.roberta_model_path = self.model_dir.joinpath('roberta-model')

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if self.device.type == 'cuda':
            self.setting_cuda()

        print(f"Using device: {self.device}")
        self.label_encoder = LabelEncoder()

    def setting_cuda(self):
        # torch.cuda.set_per_process_memory_fraction(0.9)
        pass

    def set_base_model(self, num_labels):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        self.model.to(self.device)
        print("Set standard tokenizer and ROBERTA")

    def train(self, train_df: pd.DataFrame):
        if self.model_dir.exists():
            remove_all_files_and_subdirectories_in_folder(self.model_dir)
        corpus = get_corpus(train_df)

        # Encode assignees
        assignee_ids = Database.extract_assignee_ids(train_df)
        labels = self.label_encoder.fit_transform(assignee_ids)
        num_labels = len(self.label_encoder.classes_)

        # Set up the model with the correct number of labels
        self.set_base_model(num_labels)

        optimizer = AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        train_loader = DataLoader(list(zip(corpus, labels)), batch_size=self.BATCH_SIZE, shuffle=True)

        for epoch in range(self.EPOCHS):
            epoch_loss = 0
            self.model.train()
            for texts, batch_labels in train_loader:
                preprocessed_texts = [preprocess_text(text) for text in texts]
                inputs = self.tokenize(preprocessed_texts)

                if not isinstance(batch_labels, torch.Tensor):
                    batch_labels = torch.tensor(batch_labels)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                outputs = self.model(**inputs, labels=batch_labels)
                loss = outputs.loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.EPOCHS}, Loss: {epoch_loss / len(train_loader)}")

        self.save_models()

    def save_models(self):
        joblib.dump(self.label_encoder, self.label_encoder_path)
        self.tokenizer.save_pretrained(self.tokenizer_path)
        self.model.save_pretrained(self.roberta_model_path)
        print(f"Models saved to {self.model_dir}")

    def load_models(self):
        if self.label_encoder_path.exists():
            self.label_encoder = joblib.load(self.label_encoder_path)
        else:
            raise FileNotFoundError("Label encoder not found")

        if self.tokenizer_path.exists() and self.roberta_model_path.exists():
            self.tokenizer = RobertaTokenizer.from_pretrained(self.tokenizer_path)
            self.model = RobertaForSequenceClassification.from_pretrained(self.roberta_model_path)
            self.model.to(self.device)
            print("Loaded tokenizer and ROBERTA model")
        else:
            raise FileNotFoundError("Tokenizer or RoBERTa model not found")

        print(f"Models loaded from {self.model_dir}")

    def tokenize(self, corpus: List[str]):
        inputs = self.tokenizer(
            corpus,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict_assignees(self, issue_id: int, top_n: int = 5,
                          getter: Callable[[int], pd.DataFrame] = Database.get_issues_by_id) -> List[str]:
        self.load_models() # make sure is loaded before evaluating
        issue_df = getter(issue_id)

        query_corpus = get_query_corpus(issue_df)

        self.model.eval()

        inputs = self.tokenize([query_corpus])

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the probabilities and top N predictions
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:top_n]
        assignee_ids = self.label_encoder.inverse_transform(top_indices)
        return assignee_ids.tolist()

    def predict_assignees_by_issue_number(self, number_id: int, top_n: int = 5):
        return self.predict_assignees(number_id, top_n, Database.get_issues_by_number)


    def evaluate(self, test_df: pd.DataFrame) -> float:
        self.load_models() # make sure is loaded before evaluating

        test_corpus = get_corpus(test_df)
        test_corpus = [preprocess_text(text) for text in test_corpus]

        test_labels = self.label_encoder.transform(Database.extract_assignee_ids(test_df))
        self.model.eval()
        predictions = []

        # Tokenize and predict in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(test_corpus), self.BATCH_SIZE), desc="Evaluating"):
                batch_texts = test_corpus[i:i + self.BATCH_SIZE]
                inputs = self.tokenize(batch_texts)
                outputs = self.model(**inputs)
                logits = outputs.logits
                current_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(current_predictions)
        accuracy = np.mean(np.array(predictions) == test_labels)
        return accuracy



if __name__ == '__main__':
    models_dir = get_model_dir()
    predictor_all = Predictor(models_dir)
    df = Database.get_train_set()
    predictor_all.train(df)
    test_df = Database.get_test_set()
    accuracy = predictor_all.evaluate(test_df)
    print(f"Test Accuracy predictor all: {accuracy * 100:.2f}%")

    models_dir = get_models_recenet_dir()
    predictor_recent = Predictor(models_dir)
    predictor_recent.train(df)
    accuracy = predictor_recent.evaluate(test_df)
    print(f"Test Accuracy predictor recent: {accuracy * 100:.2f}%")




