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
    """
    Creates contingency table for predictions from two models.
    """
    table = np.zeros(shape=(2, 2))
    for y_first, y_second in zip(y_first_model, y_second_model):
        table[y_first, y_second] += 1
    return table


def evaluate_two_models(first_model, second_model, data_first, data_second, device, criterion,
                        num_labels=2, features=False, batch_size=32):
    """
    Evaluates two models used for later testing. Returns a contingency table from the predictions.
    """
    loss_1, acc_1, conf_matrix1, y_first = evaluate(first_model, data_first, device, criterion, num_labels,
                                                    features=False, return_predictions=True)

    loss_2, acc_2, conf_matrix2, y_second = evaluate(second_model, data_second, device, criterion, num_labels,
                                                     features, return_predictions=True)

    acc_1 = acc_1 / len(data_first) / batch_size
    acc_2 = acc_2 / len(data_second) / batch_size

    print(f"[Model 1]: loss: {loss_1:.4f}, acc: {100*acc_1:.2f}%")
    print(f"[Model 2]: loss: {loss_2:.4f}, acc: {100*acc_2:.2f}%")

    table = create_contingency_table(y_first, y_second)
    return table


def make_tests(table, exact, correction, alpha=0.05):
    stats = mcnemar(table=table, exact=exact, correction=correction)
    print(stats)
    if stats.pvalue < alpha:
        print("P-value is less than alpha. Different proportions of error. H0 is rejected.")
    else:
        print("P-value is greater than alpha. Same proportions of error. H0 is not rejected.")



