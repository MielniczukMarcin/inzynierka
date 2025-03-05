import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trenuje klasyfikatory i oblicza metryki ewaluacyjne, używając GridSearchCV do strojenia hiperparametrów.

    Returns:
        DataFrame z wynikami klasyfikacji.
    """

    # Definicja hiperparametrów do przeszukania
    param_grid = {
        "k-NN": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        },
        "Random Forest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "k-NN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=1.0),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    }

    results = []
    for name, clf in classifiers.items():
        start_time = time.time()

        # Jeśli model ma zdefiniowane hiperparametry, zastosuj GridSearchCV
        if name in param_grid:
            grid_search = GridSearchCV(clf, param_grid[name], cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
            print(f"Najlepsze parametry dla {name}: {grid_search.best_params_}")
        else:
            clf.fit(X_train, y_train)

        end_time = time.time()
        accuracy, precision, recall, f1 = evaluate_model(clf, X_test, y_test)
        exec_time = end_time - start_time

        results.append([name, accuracy, precision, recall, f1, exec_time])

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])
    return df_results
