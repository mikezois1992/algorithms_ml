from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

class SVMClassifier:
    def __init__(self):
        self.model = OneVsRestClassifier(SVC(probability=True))

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def decision_function(self, X_test):
        return self.model.decision_function(X_test)