import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error


# Classe responsável por extrair os dados do csv e gerar um DataFrame Pandas com os Resultados
class IrisDataExtractor:
    def __init__(self):
        self.dataset = pd.read_csv(self.data_url, header=None)

    @property
    def data_url(self):
        return 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    def get_dataset(self):
        return self.dataset


class IrisPerceptron:
    def __init__(self, number_of_attributes, class_labels, innitial_weight, innitial_alpha):
        self.weights = [innitial_alpha]
        self.weights = np.append(self.weights, [innitial_weight for _ in range(number_of_attributes)], axis=0)

        self.misclassify_record = []
        self.squared_errror = []

        self._label_map = {
            1: class_labels[0],
            -1: class_labels[1]
        }
        self._reversed_label_map = {
            class_labels[0]: 1,
            class_labels[1]: -1
        }

    def _linear_combination(self, sample):
        return np.inner(sample, self.weights[1:])

    def train(self, samples, labels, max_iterator=100):
        transferred_labels = [self._reversed_label_map[index] for index in labels]

        for _ in range(max_iterator):
            misclassifies = 0

            for sample, target in zip(samples, transferred_labels):
                linear_combination = self._linear_combination(sample)
                update = target - np.where(linear_combination >= 0.0, 1, -1)

                self.weights[1:] += np.multiply(update, sample)
                self.weights[0] += update

                misclassifies += int(update != 0.0)

            # if misclassifies == 0:
            #     break

            self.misclassify_record.append(misclassifies)

            self.calculate_squared_error(samples, labels)

    def calculate_squared_error(self, samples, labels):
        transferred_labels = [self._reversed_label_map[index] for index in labels]

        actual_predict = self.classify(samples)
        transferred_actual_predict = [self._reversed_label_map[index] for index in actual_predict]

        self.squared_errror.append(mean_squared_error(transferred_labels, transferred_actual_predict))

    def classify(self, new_data):
        predicted_result = np.where((self._linear_combination(new_data) + self.weights[0]) >= 0.0, 1, -1)

        return [self._label_map[item] for item in predicted_result]



#Inicio da análise de dados
extractor = IrisDataExtractor()
IRIS_DATA = extractor.get_dataset()

SETOSA_LABEL = IRIS_DATA.iloc[0:40, 4].values
VERSICOLOR_LABEL = IRIS_DATA.iloc[50:90, 4].values
VIRGINICA_LABEL = IRIS_DATA.iloc[100:140, 4].values

SETOSA_VERSICOLOR_TRAINING_LABEL = np.append(SETOSA_LABEL, VERSICOLOR_LABEL)
SETOSA_VIRGINICA_TRAINING_LABEL = np.append(SETOSA_LABEL, VIRGINICA_LABEL)
VERSICOLOR_VIRGINICA_TRAINING_LABEL = np.append(VERSICOLOR_LABEL, VIRGINICA_LABEL)


SETOSA_DATA = IRIS_DATA.iloc[0:40, [0, 1, 2, 3]].values
VERSICOLOR_DATA = IRIS_DATA.iloc[50:90, [0, 1, 2, 3]].values
VIRGINICA_DATA = IRIS_DATA.iloc[100:140, [0, 1, 2, 3]].values

#Criação das amostras de treinamento
SETOSA_VERSICOLOR_TRAINING_DATA = np.append(SETOSA_DATA, VERSICOLOR_DATA, axis=0)
SETOSA_VIRGINICA_TRAINING_DATA = np.append(SETOSA_DATA, VIRGINICA_DATA, axis=0)
VERSICOLOR_VIRGINICA_TRAINING_DATA = np.append(VERSICOLOR_DATA, VIRGINICA_DATA, axis=0)

#Ceiação das amostras de teste
SETOSA_TEST = IRIS_DATA.iloc[40:50, [0, 1, 2, 3]].values
VERSICOLOR_TEST = IRIS_DATA.iloc[90:100, [0, 1, 2, 3]].values
VIRGINICA_TEST = IRIS_DATA.iloc[140:150, [0, 1, 2, 3]].values
TEST = np.append(SETOSA_TEST, VERSICOLOR_TEST, axis=0)
TEST = np.append(TEST, VIRGINICA_TEST, axis=0)

#Criação das amostras de validação
SETOSA_VERIFY = IRIS_DATA.iloc[40:50, 4].values
VERSICOLOR_VERIFY = IRIS_DATA.iloc[90:100, 4].values
VIRGINICA_VERIFY = IRIS_DATA.iloc[140:150, 4].values
VERIFY = np.append(SETOSA_VERIFY, VERSICOLOR_VERIFY)
VERIFY = np.append(VERIFY, VIRGINICA_VERIFY)

if __name__ == '__main__':
    # Instancia o Primeiro Perceptron
    perceptron_setosa_versicolor = IrisPerceptron(number_of_attributes=2,
                                                  class_labels=('Iris-setosa', 'Iris-versicolor'), innitial_alpha=0.0,
                                                  innitial_weight=0.0)
    # Treina o modelo
    perceptron_setosa_versicolor.train(SETOSA_VERSICOLOR_TRAINING_DATA, SETOSA_VERSICOLOR_TRAINING_LABEL)

    # Instancia o Segundo Perceptron
    perceptron_setosa_virginica = IrisPerceptron(number_of_attributes=2, class_labels=('Iris-setosa', 'Iris-virginica'),
                                                 innitial_alpha=0.0, innitial_weight=0.0)
    # Treina o Modelo
    perceptron_setosa_virginica.train(SETOSA_VIRGINICA_TRAINING_DATA, SETOSA_VIRGINICA_TRAINING_LABEL)

    # Instancia o Terceiro Perceptron
    perceptron_versicolor_virginica = IrisPerceptron(number_of_attributes=2,
                                                     class_labels=('Iris-versicolor', 'Iris-virginica'),
                                                     innitial_alpha=0.0, innitial_weight=0.0)
    # Treina o modelo
    perceptron_versicolor_virginica.train(VERSICOLOR_VIRGINICA_TRAINING_DATA, VERSICOLOR_VIRGINICA_TRAINING_LABEL)

    # Executa todos os classificadores
    predict_target_1 = perceptron_setosa_versicolor.classify(TEST)
    predict_target_2 = perceptron_setosa_virginica.classify(TEST)
    predict_target_3 = perceptron_versicolor_virginica.classify(TEST)

    overall_predict_result = []
    for item in zip(predict_target_1, predict_target_2, predict_target_3):
        unique, counts = np.unique(item, return_counts=True)
        temp_result = (zip(unique, counts))
        # Ordena pelos valores e seleciona os mais "votados"
        overall_predict_result.append(
            sorted(temp_result, reverse=True, key=lambda tup: tup[1])[0][0])

    print(overall_predict_result)

    # Confere os resultados (estava sendo utlizado antes de inserir os auxiliares da sklearn
    misclassified = 0
    for predict, verify in zip(overall_predict_result, VERIFY):
        if predict != verify:
            misclassified += 1
    print("The number of misclassified: " + str(misclassified))

    print("Accuracy score: " + str(accuracy_score(VERIFY, overall_predict_result)))
    print("\nConfusion matrix: \n" + str(confusion_matrix(VERIFY, overall_predict_result)))
    print("\nClassification report: \n" + str(classification_report(VERIFY, overall_predict_result)))

    plt.plot(perceptron_setosa_versicolor.squared_errror, color='blue', label='Erro Quadrático Setosa Versicolor')
    plt.plot(perceptron_versicolor_virginica.squared_errror, color='green',
             label='Erro Quadrático Virginica Versicolor')
    plt.plot(perceptron_setosa_virginica.squared_errror, color='red', label='Erro Quadrático Setosa Virginica')

    plt.ylabel('Erro Médio Quadrático por Épocas')
    plt.xlabel('Épocas')
    plt.title('Erro Médio Quadrático')

    plt.legend(loc='upper right')
    plt.grid(True)

    plt.show()
