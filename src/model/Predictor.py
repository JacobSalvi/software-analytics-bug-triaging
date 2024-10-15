import argparse
import ast
from pathlib import Path

from numpy import floating
from sympy.codegen import Print
from src.Database import Database
import os
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
from typing import List, Any
import joblib
from tqdm import tqdm

from src.utils.utils import remove_all_files_and_subdirectories_in_folder


class Predictor:
    BATCH_SIZE = 256
    MODEL_LOADED = False
    MODEL_DIR = Path(__file__).parent.resolve() / "../../models"
    CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.joblib')
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer')
    ROBERTA_MODEL_PATH = os.path.join(MODEL_DIR, 'roberta-model')

    def __init__(self, model_dir: Path = None, use_gpu: bool = True):
        if model_dir is not None:
            self.MODEL_DIR = model_dir
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        if self.device.type == 'cuda':
            self.setting_cuda()

        print(f"Using device: {self.device}")

        # Initialize tokenizer and model
        if os.path.exists(self.TOKENIZER_PATH) and os.path.exists(self.ROBERTA_MODEL_PATH):
            self.tokenizer = RobertaTokenizer.from_pretrained(self.TOKENIZER_PATH)
            self.model = RobertaModel.from_pretrained(self.ROBERTA_MODEL_PATH)
            self.model.to(self.device)
            print("Loaded tokenizer and RoBERTa model from disk")
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
            self.model.to(self.device)
            print("Loaded pre-trained tokenizer and RoBERTa model from the package")

        # Initialize label encoder and classifier
        self.label_encoder = LabelEncoder()
        self.classifier =  LogisticRegression(max_iter=100000, tol=1e-4, solver='lbfgs', verbose=1)

        # Load classifier and label encoder if exists
        if os.path.exists(self.CLASSIFIER_PATH) and os.path.exists(self.LABEL_ENCODER_PATH):
            self.load_models()
            self.MODEL_LOADED = True

    def setting_cuda(self):
        torch.cuda.set_per_process_memory_fraction(0.9)

    def preprocess_text(self, text: str) -> str:
        return text.lower().strip()

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        self.model.eval()  # Set model to evaluation mode
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + batch_size]
                preprocessed_texts = [self.preprocess_text(text) for text in batch_texts]
                inputs = self.tokenizer(
                    preprocessed_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                outputs = self.model(**inputs)
                # Use the [CLS] token's embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        return np.vstack(embeddings)

    def train(self, batch_size: int = BATCH_SIZE):
        if os.path.exists(self.MODEL_DIR):
            remove_all_files_and_subdirectories_in_folder(self.MODEL_DIR)

        # Load training data
        train_df = Database.get_train_set()

        print(f"Training on {len(train_df)} issues with {len(train_df["assignee"].value_counts())} assignees")

        # Create corpus
        corpus = (train_df['title'] + ' ' + train_df['body']).tolist()
        train_embeddings = self.get_embeddings(corpus, batch_size)

        # Encode assignees, use assignees ids as labels
        assignee_ids = Database.extract_assignee_ids(train_df)
        # labels = self.label_encoder.fit_transform(train_df['assignee'])
        labels = self.label_encoder.fit_transform(assignee_ids)

        # train classifier
        self.classifier.fit(train_embeddings, labels)
        self.save_models()
        self.MODEL_LOADED = True

    def save_models(self):
        joblib.dump(self.classifier, self.CLASSIFIER_PATH)
        joblib.dump(self.label_encoder, self.LABEL_ENCODER_PATH)

        self.tokenizer.save_pretrained(self.TOKENIZER_PATH)
        self.model.save_pretrained(self.ROBERTA_MODEL_PATH)
        print(f"Models saved to {self.MODEL_DIR}")

    def load_models(self):
        if os.path.exists(self.CLASSIFIER_PATH) and os.path.exists(self.LABEL_ENCODER_PATH):
            self.classifier = joblib.load(self.CLASSIFIER_PATH)
            self.label_encoder = joblib.load(self.LABEL_ENCODER_PATH)
        else:
            raise FileNotFoundError("Classifier or label encoder not found")

        if os.path.exists(self.TOKENIZER_PATH) and os.path.exists(self.ROBERTA_MODEL_PATH):
            self.tokenizer = RobertaTokenizer.from_pretrained(self.TOKENIZER_PATH)
            self.model = RobertaModel.from_pretrained(self.ROBERTA_MODEL_PATH)
            self.model.to(self.device)
        else:
            raise FileNotFoundError("Tokenizer or RoBERTa model not found")
        Print(f"Models loaded from {self.MODEL_DIR}")
        self.MODEL_LOADED = True

    def predict_assignees(self, issue_id: int, top_n: int = 5) -> List[str]:
        if not self.MODEL_LOADED:
            raise NotFittedError("Models are not loaded. Train or load models before making predictions")
        try:
            issue_df = Database.get_issues_by_id(issue_id)
        except ValueError as e:
            raise e

        issue_df.fillna('', inplace=True)
        query_corpus = issue_df.iloc[0]['title'] + ' ' + issue_df.iloc[0]['body']
        embedding = self.get_embeddings([query_corpus])

        probs = self.classifier.predict_proba(embedding)[0]
        top_indices = np.argsort(probs)[::-1][:top_n]
        assignees = self.label_encoder.inverse_transform(top_indices)

        return assignees.tolist()

    def evaluate(self, test_df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> floating[Any]:
        if not self.MODEL_LOADED:
            raise NotFittedError("Models are not loaded. Train or load models before making predictions")
        test_df = test_df.dropna(subset=['title', 'body', 'assignee'])

        test_texts = (test_df['title'] + ' ' + test_df['body']).tolist()
        test_embeddings = self.get_embeddings(test_texts, batch_size)
        test_labels = self.label_encoder.transform(Database.extract_assignee_ids(test_df))

        predictions = self.classifier.predict(test_embeddings)
        accuracy = np.mean(predictions == test_labels)
        return accuracy

