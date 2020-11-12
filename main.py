#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Decision Tree Implementation from scratch using Iris Dataset
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, fbeta_score, confusion_matrix
from ID3 import ID3DecisionTree
from utils import *


def main():
    x, y = load_dataset()

    acc_dict = {}
    f1_beta_half_dict = {}
    f1_beta_1_dict = {}
    f1_beta_2_dict = {}
    roc_dict = {}
    for bins in [5, 10, 15, 20]:
        accs = []
        f1_beta_half = []
        f1_beta_1 = []
        f1_beta_2 = []
        dt = ID3DecisionTree(nbins=bins)
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
            tree = dt.fit(X_train, y_train)
            y_hat = dt.predict(X_test)
            accs.append(balanced_accuracy_score(y_test, y_hat))
            true = np.where(y_test == 'setosa', 0, 1)
            pred = np.where(y_hat == 'setosa', 0, 1)
            f1_beta_half.append(fbeta_score(true, pred, beta=0.5))
            f1_beta_1.append(fbeta_score(true, pred, beta=1))
            f1_beta_2.append(fbeta_score(true, pred, beta=2))
        acc_dict[bins] = accs
        roc_dict[bins] = get_fpr_tpr(true, pred)
        f1_beta_half_dict[bins] = f1_beta_half
        f1_beta_1_dict[bins] = f1_beta_1
        f1_beta_2_dict[bins] = f1_beta_2

    print(print_accuracies(acc_dict))
    plot_accuracies(acc_dict, 'Decision Tree Accuracy')
    # plot_f1(f1_beta_half_dict, 'Decision Tree F1 Score (Beta = 0.5)')
    # plot_f1(f1_beta_1_dict, 'Decision Tree F1 Score (Beta = 1)')
    # plot_f1(f1_beta_2_dict, 'Decision Tree F1 Score (Beta = 2)')
    plot_roc(roc_dict, 'Decision Tree ROC Curve')


def test():
    x, y = load_dataset()
    dt = ID3DecisionTree(nbins=5)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    tree = dt.fit(X_train, y_train)
    y_hat = dt.predict(X_test)
    true = np.where(y_test == 'setosa', 0, 1)
    pred = np.where(y_hat == 'setosa', 0, 1)
    print(get_tpr_fpr(true, pred))


if __name__ == '__main__':
    main()
    # test()
