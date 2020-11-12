#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Decision Tree Implementation from scratch using Iris Dataset
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix


def load_dataset():
    iris = datasets.load_iris()
    x = iris.data
    y = ((iris.target != 0) * 1).reshape(len(x), 1)
    y = np.where(y > 0, 'non-setosa', 'setosa')

    return x, y


def print_accuracies(acc_dict):
    for bin in acc_dict:
        print('Bins: ', bin)
        print('\t Accuracies: ', acc_dict[bin])
        print('\t Min Acc: ', min(acc_dict[bin]))
        print('\t Max Acc: ', max(acc_dict[bin]))
        print('\t Avg Acc: ', np.average(acc_dict[bin]))


def plot_accuracies(acc_dict, title):
    for bin in acc_dict:
        plt.plot(acc_dict[bin], label='bins='+str(bin))
    plt.ylabel('Accuracy')
    plt.xlabel('Random Samples')
    plt.xticks(range(len(acc_dict)+1))
    plt.legend()
    plt.title(title)
    plt.savefig('figs/' + title + '.png')
    plt.clf()


def plot_f1(f1_dict, title):
    for bin in f1_dict:
        plt.plot(f1_dict[bin], label='bins='+str(bin))
    plt.ylabel('F1 Score')
    plt.xlabel('Random Samples')
    plt.xticks(range(len(f1_dict)+1))
    plt.legend()
    plt.title(title)
    plt.savefig('figs/' + title + '.png')
    plt.clf()


def get_fpr_tpr(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return 1-specificity, sensitivity


def plot_roc(roc_dict, title):
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'r--', label='Random Guess')
    for bin in roc_dict:
        plt.plot(roc_dict[bin][0], roc_dict[bin][1], marker='o', label='bin'+str(bin))
    plt.legend()
    plt.xlabel('FPR (1-specificity)')
    plt.ylabel('TPR (sensitivity)')
    plt.title(title)
    plt.savefig('figs/' + title + '.png')
