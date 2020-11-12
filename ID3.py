#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
ID3 Algorithm Implementation for decision tree
'''

import numpy as np


class ID3DecisionTree:
    def __init__(self, nbins=10):
        self.nbins = nbins
        self.tree = {}

    def purity_check(self, labels):
        unique_classes = np.unique(labels)
        if len(unique_classes) == 1:
            return True
        else:
            return False

    def classify_data(self, labels):
        unique_classes, counts_unique_classes = np.unique(labels, return_counts=True)
        index = counts_unique_classes.argmax()
        classification = unique_classes[index]

        return classification

    def get_potential_splits(self, data, nbins):
        potential_splits = {}
        _, n_columns = data.shape
        for col in range(n_columns):
            potential_splits[col] = np.histogram(data[:, col], bins=nbins)[1]
        return potential_splits

    def split_data(self, data, labels, split_column, split_value):
        below_ind = np.where(data[:, split_column] <= split_value)
        above_ind = np.where(data[:, split_column] > split_value)
        data_below = data[below_ind]
        data_above = data[above_ind]
        labels_below = labels[below_ind]
        labels_above = labels[above_ind]
        return data_below, labels_below, data_above, labels_above

    def calculate_entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = sum(probs * -np.log2(probs))

        return entropy

    def calculate_total_entropy(self, labels_below, labels_above):
        num_total = len(labels_below) + len(labels_above)
        p_below = len(labels_below) / num_total
        p_above = len(labels_above) / num_total
        total_entropy = (p_below * self.calculate_entropy(labels_below))\
                        + (p_above * self.calculate_entropy(labels_above))

        return total_entropy

    def best_split(self, data, labels, potential_splits):
        overall_entropy = 1
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, labels_below, data_above, labels_above = self.split_data(data, labels,
                                                                                     split_column=column_index,
                                                                                     split_value=value)
                current_overall_entropy = self.calculate_total_entropy(labels_below, labels_above)

                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value

        return best_split_column, best_split_value

    def fit(self, data, labels):
        if self.purity_check(labels):
            classification = self.classify_data(labels)
            return classification

        # recursive part
        else:
            potential_splits = self.get_potential_splits(data, self.nbins)
            split_column, split_value = self.best_split(data, labels, self.get_potential_splits(data, self.nbins))
            data_below, labels_below, data_above, labels_above = self.split_data(data, labels, split_column, split_value)

            # instantiate sub-tree
            decision = "{} <= {}".format(split_column, split_value)
            sub_tree = {decision: []}

            # find answers (recursion)
            yes_answer = self.fit(data_below, labels_below)
            no_answer = self.fit(data_above, labels_above)

            sub_tree[decision].append(yes_answer)
            sub_tree[decision].append(no_answer)

            self.tree = sub_tree
            return sub_tree

    def predict_one(self, test_data, tree):
        decision = list(tree.keys())[0]
        feature_col, comparison_operator, value = decision.split()
        feature_col, value = int(feature_col), float(value)

        # check decision
        if test_data[feature_col] <= value:
            answer = tree[decision][0]
        else:
            answer = tree[decision][1]

        # base case
        if not isinstance(answer, dict):
            return answer

        # recursive part
        else:
            residual_tree = answer
            return self.predict_one(test_data, residual_tree)

    def predict(self, test_data):
        preds = []
        for row in test_data:
            preds.append(self.predict_one(row, self.tree))
        return np.array(preds)
