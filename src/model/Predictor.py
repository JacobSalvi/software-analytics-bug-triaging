import os
from pathlib import Path
from typing import List
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
from src.utils.utils import remove_all_files_and_subdirectories_in_folder


class Predictor:
    BATCH_SIZE = 32
    MAX_ITERATION = 5
    LEARNING_RATE = 2e-5
    TOLERANCE = 1e-5

    MODEL_DIR = utils.get_model_dir()
    LABEL_ENCODER_PATH = MODEL_DIR.joinpath( 'label_encoder.joblib')
    TOKENIZER_PATH = MODEL_DIR.joinpath('tokenizer')
    ROBERTA_MODEL_PATH = MODEL_DIR.joinpath('roberta-model')

    def __init__(self, model_dir: Path = None, use_gpu: bool = True,
                 batch_size: int = BATCH_SIZE, max_iteration: int = MAX_ITERATION,
                 tolerance: float = TOLERANCE):
        self.MAX_ITERATION = max_iteration
        self.TOLERANCE = tolerance
        self.BATCH_SIZE = batch_size

        if model_dir is not None:
            self.MODEL_DIR = model_dir
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if self.device.type == 'cuda':
            self.setting_cuda()

        print(f"Using device: {self.device}")

        # Initialize label encoder
        self.label_encoder = LabelEncoder()

    def setting_cuda(self):
        pass  # Commented out for now
        # torch.cuda.set_per_process_memory_fraction(0.9)

    def preprocess_text(self, text: str) -> str:
        return text.lower().strip()

    def get_corpus(self, train_df: pd.DataFrame) -> List[str]:
        return (train_df['title'] + ' ' + train_df['body'] + ' ' + train_df['labels']).tolist()

    def get_assignee_ids(self, train_df: pd.DataFrame) -> List[int]:
        return Database.extract_assignee_ids(train_df)

    def set_base_model(self, num_labels):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        self.model.to(self.device)
        print("Loaded pre-trained tokenizer and RoBERTa model with classification head")

    def train(self, batch_size: int = BATCH_SIZE):
        if Path.exists(self.MODEL_DIR):
            remove_all_files_and_subdirectories_in_folder(self.MODEL_DIR)
        train_df = Database.get_train_set().copy()
        corpus = self.get_corpus(train_df)

        # Encode assignees
        assignee_ids = Database.extract_assignee_ids(train_df)
        labels = self.label_encoder.fit_transform(assignee_ids)
        num_labels = len(self.label_encoder.classes_)

        # Now set up the model with the correct number of labels
        self.set_base_model(num_labels)

        optimizer = AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        train_loader = DataLoader(list(zip(corpus, labels)), batch_size=batch_size, shuffle=True)

        for epoch in range(self.MAX_ITERATION):
            epoch_loss = 0
            self.model.train()
            for texts, batch_labels in train_loader:
                preprocessed_texts = [self.preprocess_text(text) for text in texts]
                inputs = self.tokenizer(
                    preprocessed_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                # Move inputs to the device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Ensure batch_labels is a tensor and move to device
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

            print(f"Epoch {epoch + 1}/{self.MAX_ITERATION}, Loss: {epoch_loss / len(train_loader)}")

        self.save_models()
        self.MODEL_LOADED = True

    def save_models(self):
        joblib.dump(self.label_encoder, self.LABEL_ENCODER_PATH)
        self.tokenizer.save_pretrained(self.TOKENIZER_PATH)
        self.model.save_pretrained(self.ROBERTA_MODEL_PATH)
        print(f"Models saved to {self.MODEL_DIR}")

    def load_models(self):
        if os.path.exists(self.LABEL_ENCODER_PATH):
            self.label_encoder = joblib.load(self.LABEL_ENCODER_PATH)
        else:
            raise FileNotFoundError("Label encoder not found")

        if os.path.exists(self.TOKENIZER_PATH) and os.path.exists(self.ROBERTA_MODEL_PATH):
            self.tokenizer = RobertaTokenizer.from_pretrained(self.TOKENIZER_PATH)
            self.model = RobertaForSequenceClassification.from_pretrained(self.ROBERTA_MODEL_PATH)
            self.model.to(self.device)
            print("Loaded tokenizer and RoBERTa model from disk")
        else:
            raise FileNotFoundError("Tokenizer or RoBERTa model not found")

        print(f"Models loaded from {self.MODEL_DIR}")
        self.MODEL_LOADED = True

    def predict_assignees(self, issue_id: int, top_n: int = 5) -> List[str]:
        self.load_models()
        try:
            issue_df = Database.get_issues_by_id(issue_id)
        except ValueError as e:
            raise e

        issue_df.fillna('', inplace=True)
        query_corpus = issue_df.iloc[0]['title'] + ' ' + issue_df.iloc[0]['body'] + ' ' + issue_df.iloc[0]['labels']
        query_corpus = self.preprocess_text(query_corpus)
        self.model.eval()
        inputs = self.tokenizer(
            [query_corpus],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the probabilities and top N predictions
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:top_n]
        assignee_ids = self.label_encoder.inverse_transform(top_indices)

        return assignee_ids.tolist()

    def evaluate(self, test_df: pd.DataFrame) -> float:
        self.load_models()
        test_df["body"] = test_df["body"].fillna("")
        test_texts = (test_df['title'] + ' ' + test_df['body'] + ' ' + test_df['labels']).tolist()
        test_texts = [self.preprocess_text(text) for text in test_texts]
        test_labels = self.label_encoder.transform(Database.extract_assignee_ids(test_df))

        self.model.eval()
        predictions = []

        # Tokenize and predict in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(test_texts), self.BATCH_SIZE), desc="Evaluating"):
                batch_texts = test_texts[i:i + self.BATCH_SIZE]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        accuracy = np.mean(np.array(predictions) == test_labels)
        return accuracy
