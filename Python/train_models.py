import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from evaluate import evaluate_model

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trenuje klasyfikatory i oblicza metryki ewaluacyjne.

    Returns:
        DataFrame z wynikami klasyfikacji.
    """
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),  # Ograniczamy głębokość drzewa
        "k-NN": KNeighborsClassifier(n_neighbors=5, weights='distance'),  # Wybór optymalnej liczby sąsiadów
        "Naive Bayes": GaussianNB(),  # Domyślne ustawienia
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),  # Więcej drzew, ograniczona głębokość
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0),  # Standardowe parametry
        "SVM": SVC(C=1.0, kernel='rbf', gamma='scale'),  # Kernel RBF zamiast liniowego
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)  # Standardowe ustawienia
    }

    results = []
    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()

        accuracy, precision, recall, f1 = evaluate_model(clf, X_test, y_test)
        exec_time = end_time - start_time

        results.append([name, accuracy, precision, recall, f1, exec_time])

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])
    return df_results
