import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

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
        "AdaBoost": AdaBoostClassifier()
    }

    results = []
    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end_time = time.time()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        exec_time = end_time - start_time

        results.append([name, accuracy, precision, recall, f1, exec_time])

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])
    return df_results
