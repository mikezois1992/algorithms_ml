import pandas as pd
from sklearn.manifold import TSNE

class TSNEHandler:
    def __init__(self, data):
        self.data = data

    def perform_tsne(self):
        tsne = TSNE(n_components=2)
        tsne_result = tsne.fit_transform(self.data)
        tsne_df = pd.DataFrame(data=tsne_result, columns=["t-SNE1", "t-SNE2"])
        return tsne_df
