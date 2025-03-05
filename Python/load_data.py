from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_dataset():
    """
    Wczytuje zbiór danych Iris i dzieli go na zbiór treningowy i testowy.

    Returns:
        X_train, X_test, y_train, y_test - podzielone dane
    """
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
