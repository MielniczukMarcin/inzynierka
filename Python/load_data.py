from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

def load_dataset(dataset_name="iris"):
    """
    Wczytuje wybrany zbiór danych i dzieli go na zbiór treningowy i testowy.

    Args:
        dataset_name: Nazwa zbioru danych ("iris", "wine", "breast_cancer").

    Returns:
        X_train, X_test, y_train, y_test - podzielone dane
    """
    datasets = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer
    }

    if dataset_name not in datasets:
        raise ValueError(f"Niepoprawna nazwa zbioru danych: {dataset_name}. Dostępne: {list(datasets.keys())}")

    data = datasets[dataset_name]()  # Ładujemy wybrany zbiór
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
