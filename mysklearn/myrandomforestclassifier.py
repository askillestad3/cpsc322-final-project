
from myclassifiers import MyDecisionTreeClassifier

class RandomForestDecisionTree(MyDecisionTreeClassifier):
    def __init__(self, F: int | None=None):
        super().__init__()
        self.F = F

    def fit(self, X_train: list[list], y_train: list):
        self.X_train = X_train
        self.y_train = y_train


class MyRandomForestClassifier():
    def __init__(self, N: int, M: int | None=None, F: int | None=None):
        self.N = N
        self.M = M if M is not None else N
        self.F = F
        self.X_train = None
        self.y_train = None
        self.trees: list[RandomForestDecisionTree] = []

    def fit(self, X_train: list[list], y_train: list):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test) -> list:
        return []

