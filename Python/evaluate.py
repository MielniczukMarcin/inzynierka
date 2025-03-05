from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


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


def compare_results(df_results, save_to_csv=False, filename="results.csv"):
    """
    Porównuje wyniki klasyfikatorów, sortując po dokładności (Accuracy).

    Args:
        df_results: DataFrame z wynikami klasyfikacji.
        save_to_csv: Czy zapisać wyniki do pliku CSV.
        filename: Nazwa pliku CSV (domyślnie "results.csv").
    """
    df_sorted = df_results.sort_values(by="Accuracy", ascending=False)  # Sortowanie po Accuracy
    print("\n📊 Wyniki klasyfikacji (posortowane wg Accuracy):")
    print(df_sorted)

    if save_to_csv:
        df_sorted.to_csv(filename, index=False)
        print(f"\n📁 Wyniki zapisane do pliku: {filename}")
