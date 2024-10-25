from typing import List
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.colors as pc
from src.Database import Database
from src.model.Predictor import Predictor


def plot_tsne(embeddings: np.ndarray, labels: List[int], hover_texts: List[str]):
    embeddings = normalize(embeddings)

    n_samples = embeddings.shape[0]
    perplexity = min(30, n_samples - 1) if n_samples >= 5 else max(1, n_samples - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=5000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    githubid_labels = []
    for label in labels:
        user_info = Database.get_user_by_id(label)
        if isinstance(user_info, dict) and 'login' in user_info:
            githubid_labels.append(user_info['login'])
        else:
            githubid_labels.append('Unknown')

    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': pd.Categorical(githubid_labels),
        'hover_text': hover_texts
    })

    unique_labels = df['label'].unique()
    color_palette = pc.qualitative.Plotly if len(unique_labels) <= 10 else pc.qualitative.Light24

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='label',
        hover_name='hover_text',
        title="t-SNE of Bug Report Embeddings by Assignees (GitHub IDs)",
        color_discrete_sequence=color_palette
    )

    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title="Assignees (GitHub IDs)"
    )

    fig.show()


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

    sample_bug_reports = get_sample_bug_reports(n=1000)
    embeddings, _ = predictor.get_data_embeddings(sample_bug_reports)
    # print(sample_bug_reports['labels'])
    labels = []
    hover_texts = []
    for idx, row in sample_bug_reports.iterrows():
        issue_id = row['id']
        title = row['labels']
        hover_texts.append(f"ID: {issue_id}, Title: {title}")

        try:
            top_assignees = predictor.predict_assignees(issue_id, top_n=1)
            if top_assignees and top_assignees[0]:
                labels.append(top_assignees[0])
            else:
                labels.append('Unknown')
        except Exception as e:
            print(f"Error predicting assignee for issue ID {issue_id}: {e}")
            labels.append('Unknown')

    print("Labels and hover texts retrieved.")
    plot_tsne(embeddings, labels, hover_texts)


if __name__ == "__main__":
    main()