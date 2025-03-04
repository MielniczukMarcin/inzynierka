import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- 1. Wczytanie danych ---
data = load_iris()
X, y = data.data, data.target

# --- 2. Podział na zbiór treningowy i testowy ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Lista klasyfikatorów ---
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "k-NN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# --- 4. Trenowanie i ewaluacja modeli ---
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

# --- 5. Konwersja wyników do DataFrame ---
df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])

# --- 6. Wyświetlenie tabeli wyników ---
print(df_results)


# --- 7. Wizualizacja wyników ---
plt.figure(figsize=(10, 5))
df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar")
plt.title("Porównanie klasyfikatorów")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
