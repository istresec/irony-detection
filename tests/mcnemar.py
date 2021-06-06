from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import matthews_corrcoef
from util.utils import evaluate, calculate_statistics

import numpy as np


def create_contingency_table(y_first_model, y_second_model):
    """
    Creates contingency table for predictions from two models.
    """
    table = np.zeros(shape=(2, 2))
    for y_first, y_second in zip(y_first_model, y_second_model):
        table[y_first, y_second] += 1
    return table


def evaluate_two_models(first_model, second_model, data_first, data_second, device, criterion, y_gt1, y_gt2,
                        num_labels=2, features=False, batch_size=32):
    """
    Evaluates two models used for later testing. Returns a contingency table from the predictions.
    """
    loss_1, acc_1, conf_matrix1, y_first = evaluate(first_model, data_first, device, criterion, num_labels,
                                                    features=False, return_predictions=True)
    p_1, r_1, f1_1 = calculate_statistics(conf_matrix1)
    m_1 = matthews_corrcoef(y_gt1, y_first)

    loss_2, acc_2, conf_matrix2, y_second = evaluate(second_model, data_second, device, criterion, num_labels,
                                                     features, return_predictions=True)
    p_2, r_2, f1_2 = calculate_statistics(conf_matrix2)
    m_2 = matthews_corrcoef(y_gt2, y_second)

    acc_1 = acc_1 / len(data_first) / batch_size
    acc_2 = acc_2 / len(data_second) / batch_size

    print(f"[Model 1]: loss: {loss_1:.4f}, acc: {acc_1:.4f}, p: {p_1:.4f}, r: {r_1:.4f}, f1: {f1_1:.4f}, m: {m_1:.4f}")
    print(f"[Model 2]: loss: {loss_2:.4f}, acc: {acc_2:.4f}, p: {p_2:.4f}, r: {r_2:.4f}, f1: {f1_2:.4f}, m: {m_2:.4f}")

    table = create_contingency_table(y_first, y_second)
    return table


def make_tests(table, exact, correction, alpha=0.05):
    stats = mcnemar(table=table, exact=exact, correction=correction)
    print(stats)
    if stats.pvalue < alpha:
        print("P-value is less than alpha. Different proportions of error. H0 is rejected.")
    else:
        print("P-value is greater than alpha. Same proportions of error. H0 is not rejected.")



