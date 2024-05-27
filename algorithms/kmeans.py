from sklearn.cluster import KMeans

class KMeansHandler:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def perform_kmeans(self):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans_result = kmeans.fit_predict(self.data)
        return kmeans_result
