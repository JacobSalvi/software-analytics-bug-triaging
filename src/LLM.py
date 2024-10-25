from torch.nn.functional import embedding
from transformers import RobertaTokenizer, RobertaModel
import torch
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from src.DataHandler import DataHandler
from src.Database import Database
from src.model.Predictor import Predictor


def load_llm(bug_reports) :
    predictor = Predictor()
    embeddings, _ = predictor.get_data_embeddings(bug_reports)
    return embeddings

def clustering(radius, min_samples, embeddings_np):
    
    embeddings_np_scaled = StandardScaler().fit_transform(embeddings_np)

    
    dbscan = DBSCAN(eps=radius, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings_np_scaled)

    
    return cluster_labels

def visualize(embeddings_np,cluster_labels,bug_reports):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings_np)

    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'bug_report': bug_reports,
        'cluster': cluster_labels
        }
    )

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster', 
        hover_data=['bug_report'],  
        title="Bug Report Clusters"
    )

    fig.update_layout(
        legend_title_text='Clusters',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            )
    )
    fig.show()
    



def main():
    bug_reports = Database.get_train_set()
    embeddings_np = load_llm(bug_reports)
    radius = float(input("Define the radius of clusters "))
    min_points = int(input("Define the minimum points for a cluster "))
    cluster_labels = clustering(radius,min_points,embeddings_np)
    visualize(embeddings_np,cluster_labels,bug_reports)

if __name__ == "__main__" :
    main()
