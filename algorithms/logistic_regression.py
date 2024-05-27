from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

class LogisticRegressionClassifier:
    def __init__(self):
        self.model = OneVsRestClassifier(LogisticRegression())

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def decision_function(self, X_test):
        return self.model.decision_function(X_test)