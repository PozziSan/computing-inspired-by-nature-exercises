import numpy as np


class IrisPerceptron:
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
        return np.inner(sample, self.weights)

    def train(self, samples, labels, max_iterator=10):
        transferred_labels = [
            self._reversed_label_map[index] for index in labels
        ]

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
