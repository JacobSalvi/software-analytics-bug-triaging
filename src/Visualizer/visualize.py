# visualizer.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def plot_tsne(embeddings, labels, title="t-SNE of Bug Report Embeddings"):
   
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })

    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='tab10', legend='full')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    from src.model.Predictor import Predictor

    predictor = Predictor()
    embeddings, labels = predictor.train(return_embedding=True)

    
    plot_tsne(embeddings, labels, title="t-SNE of Bug Report Embeddings by Assignees")
