import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Classe responsável por extrair os dados do csv e gerar um DataFrame Pandas com os Resultados
class WineDataExtractor:
    def __init__(self):
        self.dataset = pd.read_csv(self.data_url, header=None)

    @property
    def data_url(self):
        return 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

    def get_dataset(self):
        return self.dataset


if __name__ == '__main__':
    # inicia analise de dados
    extractor = WineDataExtractor()

    data = extractor.get_dataset()

    labels = data.values[:, 0]
    values = data.values[:, 1:13]

    #divide os dados entre treino e teste
    train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.12,
                                                                            random_state=1)

    # normaliza dos dados
    standard_scaler = StandardScaler()
    standard_scaler.fit(train_values)

    train_values = standard_scaler.transform(train_values)
    test_values = standard_scaler.transform(test_values)

    # instancia o perceptron
    perceptron_classifier = Perceptron(max_iter=1500, eta0=0.1, random_state=1)
    perceptron_classifier.fit(train_values, train_labels)
    labels_pred = perceptron_classifier.predict(test_values)

    # exibe informações da classificação
    print("Accuracy score: " + str(accuracy_score(test_labels, labels_pred)))
    print("\nConfusion matrix: \n" + str(confusion_matrix(test_labels, labels_pred)))
    print("\nClassification report: \n" + str(classification_report(test_labels, labels_pred)))

    # instancia o LR
    logistic_regression_classifier = LogisticRegression(C=100, max_iter=1500, random_state=1)
    logistic_regression_classifier.fit(train_values, train_labels)
    lr_labels_pred = logistic_regression_classifier.predict(test_values)

    # exibe informações da classificação
    print("Accuracy score: " + str(accuracy_score(test_labels, lr_labels_pred)))
    print("\nConfusion matrix: \n" + str(confusion_matrix(test_labels, lr_labels_pred)))
    print("\nClassification report: \n" + str(classification_report(test_labels, lr_labels_pred)))

    # Instancia o DTC
    decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
    decision_tree_classifier.fit(train_values, train_labels)
    dt_labels_pred = decision_tree_classifier.predict(test_values)

    # exibe informações da classificação
    print("Accuracy score: " + str(accuracy_score(test_labels, dt_labels_pred)))
    print("\nConfusion matrix: \n" + str(confusion_matrix(test_labels, dt_labels_pred)))
    print("\nClassification report: \n" + str(classification_report(test_labels, dt_labels_pred)))

    # Instancia o SVM
    support_vector_machine = SVC(C=10000, kernel='rbf', degree=3)
    support_vector_machine.fit(train_values, train_labels)
    svc_labels_pred = support_vector_machine.predict(test_values)

    # exibe informações da classificação
    print("Accuracy score: " + str(accuracy_score(test_labels, svc_labels_pred)))
    print("\nConfusion matrix: \n" + str(confusion_matrix(test_labels, svc_labels_pred)))
    print("\nClassification report: \n" + str(classification_report(test_labels, svc_labels_pred)))
