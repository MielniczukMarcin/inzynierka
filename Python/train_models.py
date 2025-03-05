import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # Nowy klasyfikator SVM
from evaluate import evaluate_model  # Import funkcji do oceny modeli

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trenuje klasyfikatory i oblicza metryki ewaluacyjne.

    Returns:
        DataFrame z wynikami klasyfikacji.
    """
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(),
        "k-NN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM": SVC(),  # Nowy klasyfikator SVM
        "Gradient Boosting": GradientBoostingClassifier()  # Nowy model boostingowy
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
