from sklearn.cluster import DBSCAN

class DBSCANHandler:
    def __init__(self, data, eps, min_samples):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples

    def perform_dbscan(self):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan_result = dbscan.fit_predict(self.data)
        return dbscan_result
