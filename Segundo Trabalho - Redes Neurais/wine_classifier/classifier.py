import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class WineDataExtractor:
    def __init__(self):
        self.dataset = pd.read_csv(self.data_url, header=None)

    @property
    def data_url(self):
        return 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

    def get_dataset(self):
        return self.dataset


class WinePerceptron:
    def __init__(self, number_of_attributes, class_labels):
        self.weights = np.zeros(number_of_attributes + 1)
        self.misclassify_record = []

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

    def train(self, samples, labels, max_iterator=500):
        transferred_labels = [self._reversed_label_map[index] for index in labels]

        for _ in range(max_iterator):
            misclassifies = 0

            for sample, target in zip(samples, transferred_labels):
                linear_combination = self._linear_combination(sample)
                update = target - np.where(linear_combination >= 0.0, 1, -1)

                self.weights[1:] += np.multiply(update, sample)
                self.weights[0] += update

                misclassifies += int(update != 0.0)

            if misclassifies == 0:
                break

            self.misclassify_record.append(misclassifies)

    def classify(self, new_data):
        predicted_result = np.where((self._linear_combination(new_data) + self.weights[0]) >= 0.0, 1, -1)

        return [self._label_map[item] for item in predicted_result]


extractor = WineDataExtractor()
WINE_DATA = extractor.get_dataset()
WINE_DATA.columns = ['Label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols', 'Flavanoids' , 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


if __name__ == '__main__':
    WINE_LABELS = WINE_DATA.values[:, 0]
    WINE_DATA_VALUES = WINE_DATA.values[:, 1:13]

    standard_scaller = StandardScaler()
    standard_scaller.fit(WINE_DATA_VALUES)

    WINE_DATA_VALUES = standard_scaller.transform(WINE_DATA_VALUES)

    FIRST_CLASS_LABELS = [str(item) for item in WINE_LABELS[0:58]]
    FIRST_CLASS_VALUES = WINE_DATA_VALUES[0:58]

    SECOND_CLASS_LABELS = [str(item) for item in WINE_LABELS[59:129]]
    SECOND_CLASS_VALUES = WINE_DATA_VALUES[59:129]

    THIRD_CLASS_LABELS = [str(item) for item in WINE_LABELS[130:]]
    THIRD_CLASS_VALUES = WINE_DATA_VALUES[130:]

    first_class_train_values, first_class_test_values, first_class_train_labels, first_class_test_labels = train_test_split(FIRST_CLASS_VALUES, FIRST_CLASS_LABELS, test_size=0.2, random_state=1)

    second_class_train_values, second_class_test_values, second_class_train_labels, second_class_test_labels = train_test_split(
        SECOND_CLASS_VALUES, SECOND_CLASS_LABELS, test_size=0.2, random_state=1)

    third_class_train_values, third_class_test_values, third_class_train_labels, third_class_test_labels = train_test_split(
        THIRD_CLASS_VALUES, THIRD_CLASS_LABELS, test_size=0.2, random_state=1)

    first_class_second_class_training_values = np.append(first_class_train_values, second_class_train_values, axis=0)
    first_class_second_class_labels = np.append(first_class_train_labels, second_class_train_labels, axis=0)

    second_class_third_class_training_values = np.append(second_class_train_values, third_class_train_values, axis=0)
    second_class_third_class_training_labels = np.append(second_class_train_labels, third_class_train_labels, axis=0)

    first_class_third_class_training_values = np.append(first_class_train_values, third_class_train_values, axis=0)
    first_class_third_class_training_labels = np.append(first_class_train_labels, third_class_train_labels, axis=0)

    first_class_second_class_classifier = WinePerceptron(number_of_attributes=12, class_labels=['1.0', '2.0'])
    first_class_second_class_classifier.train(first_class_second_class_training_values, first_class_second_class_labels)

    second_class_third_class_classifier = WinePerceptron(number_of_attributes=12, class_labels=['2.0', '3.0'])
    second_class_third_class_classifier.train(second_class_third_class_training_values, second_class_third_class_training_labels)

    first_class_third_class_classifier = WinePerceptron(number_of_attributes=12, class_labels=['1.0', '3.0'])
    first_class_third_class_classifier.train(first_class_third_class_training_values, first_class_third_class_training_labels)

    test_data = np.append(first_class_test_values, second_class_test_values, axis=0)
    test_data = np.append(test_data, third_class_test_values, axis=0)
    test_label = np.append(first_class_test_labels, second_class_test_labels, axis=0)
    test_label = np.append(test_label, third_class_test_labels, axis=0)

    first_predict_target = first_class_second_class_classifier.classify(test_data)
    second_predict_target = second_class_third_class_classifier.classify(test_data)
    third_predict_target = first_class_third_class_classifier.classify(test_data)

    overall_predict_result = []
    for item in zip(first_predict_target, second_predict_target, third_predict_target):
        unique, counts = np.unique(item, return_counts=True)
        temp_result = (zip(unique, counts))
        # Sort by values and return the class that has majority votes
        overall_predict_result.append(
            sorted(temp_result, reverse=True, key=lambda tup: tup[1])[0][0])

    misclassified = 0
    for predict, verify in zip(overall_predict_result, test_label):
        if predict != verify:
            misclassified += 1

    print(len(test_data))
    print("The number of misclassified: " + str(misclassified))
