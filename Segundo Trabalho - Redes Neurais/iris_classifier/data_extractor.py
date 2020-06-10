import pandas as pd


class IrisDataExtractor:
    def __init__(self):
        self.dataset = pd.read_csv(self.data_url, header=None)

    @property
    def data_url(self):
        return 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    def get_dataset(self):
        return self.dataset
