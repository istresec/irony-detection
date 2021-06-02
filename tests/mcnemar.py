"""
Testovi:
potrebno je ucitati dva modela - dati path do njih gdje se ucitava best model
za svaki model paralelno spremiti po train datasetu predikcije i onda na temelju toga formirati 2d tablicu
ucitavanje datseta ide kao u demos, formiranje matrice kao i confusion matrix
"""
from statsmodels.stats.contingency_tables import mcnemar
from util.utils import evaluate

import numpy as np


def create_contingency_table(y_first_model, y_second_model):
    table = np.zeros(shape=(2, 2))
    for y_first, y_second in zip(y_first_model, y_second_model):
        table[y_first, y_second] += 1
    return table


def evaluate_two_models(first_model, second_model, data_first, data_second, device, criterion,
                        num_labels=2, features=False):
    loss_2, acc_2, conf_matrix2, y_second = evaluate(second_model, data_second, device, criterion, num_labels,
                                                     features, return_predictions=True)
    loss_1, acc_1, conf_matrix1, y_first = evaluate(first_model, data_first, device, criterion, num_labels,
                                                    features, return_predictions=True)

    table = create_contingency_table(y_first, y_second)
    return table


def make_tests(table, exact, correction, alpha=0.05):
    stats = mcnemar(table=table, exact=exact, correction=correction)
    print(stats)
    if stats.pvalue < alpha:
        print("P-value is less than alpha. Different proportions of error. H0 is rejected.")
    else:
        print("P-value is greater than alpha. Same proportions of error. H0 is not rejected.")



