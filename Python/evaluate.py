from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(clf, X_test, y_test):
    """
    Oblicza metryki ewaluacyjne dla podanego modelu.

    Args:
        clf: Wytrenowany klasyfikator.
        X_test: Dane testowe.
        y_test: Etykiety testowe.

    Returns:
        accuracy, precision, recall, f1-score
    """
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, precision, recall, f1
