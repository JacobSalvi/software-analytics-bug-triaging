from transformers import RobertaTokenizer, RobertaModel
import torch
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from src.DataProvider import DataProvider


def stopRemoval(issues):
    stop = stopwords.words('english')
    issues['title'] = issues['title'].apply(lambda x : ' '.join([word for word in x.split() if word not in (stop)]))
    return issues

def load_issues() :
    issues = DataProvider.get_parsed()
    #issues = stopRemoval(issues)
    bug_reports = issues["title"].head(1000).tolist()
    return bug_reports


def load_llm(bug_reports) :
    model_name = "roberta-base"  
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    inputs = tokenizer(bug_reports, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    
        last_hidden_states = outputs.last_hidden_state  

    
        embeddings = torch.mean(last_hidden_states, dim=1) 

    embeddings_np = embeddings.cpu().numpy()
    return embeddings_np

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
        })

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
    bug_reports = load_issues()
    embeddings_np = load_llm(bug_reports)
    radius = float(input("Define the radius of clusters "))
    min_points = int(input("Define the minimum points for a cluster "))
    cluster_labels = clustering(radius,min_points,embeddings_np)
    visualize(embeddings_np,cluster_labels,bug_reports)

if __name__ == "__main__" :
    main()
