# visualizer.py
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

from src.Database import Database
from src.model.Predictor import Predictor


def plot_tsne(embeddings: np.ndarray, labels: List[str]):
    n_samples = embeddings.shape[0]

    if n_samples <= 1:
        raise ValueError("Need at least two samples for t-SNE.")
    elif n_samples < 5:
        perplexity = max(1, n_samples - 1)
    else:
        perplexity = min(30, n_samples - 1)


    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })

    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid", rc={"figure.figsize": (12, 10)})

    # Determine the number of unique labels to set the palette
    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette("hsv", len(unique_labels))

    scatter = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='label',
        palette=palette,
        legend='full',
        alpha=0.7
    )

    plt.title("t-SNE of Bug Report Embeddings by Assignees", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title='Assignees', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def get_sample_bug_reports(n: int = 10) -> pd.DataFrame:
    all_bug_reports = Database.get_test_set(Database.get_train_set())
    if len(all_bug_reports) < n:
        print(f"Requested {n} bug reports, but only {len(all_bug_reports)} available.")
        sample_bug_reports = all_bug_reports
    else:
        sample_bug_reports = all_bug_reports.sample(n=n, random_state=42)
    return sample_bug_reports

def main():
    predictor = Predictor()
    predictor.load_models()

    sample_bug_reports = get_sample_bug_reports(n=10)

    embeddings, _ = predictor.get_data_embeddings(sample_bug_reports)

    labels = []
    for idx, row in sample_bug_reports.iterrows():
        issue_id = row['id']  
        try:
            top_assignees = predictor.predict_assignees(issue_id, top_n=1)
            if top_assignees and top_assignees[0]:
                labels.append(top_assignees[0])
            else:
                labels.append('Unknown')
        except Exception as e:
            print(f"Error predicting assignee for issue ID {issue_id}: {e}")
            labels.append('Unknown')

    print("Labels retrieved.")

    plot_tsne(embeddings, labels)

if __name__ == "__main__":
    main()
