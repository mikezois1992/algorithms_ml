import pandas as pd
from sklearn.decomposition import PCA

class PCAHandler:
    def __init__(self, data):
        self.data = data

    def perform_pca(self):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.data)
        pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
        return pca_df