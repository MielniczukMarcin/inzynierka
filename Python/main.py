import matplotlib.pyplot as plt
from load_data import load_dataset
from train_models import train_and_evaluate

# --- 1. Wczytanie danych ---
X_train, X_test, y_train, y_test = load_dataset()

# --- 2. Trenowanie i ewaluacja modeli ---
df_results = train_and_evaluate(X_train, X_test, y_train, y_test)

# --- 3. Wyświetlenie tabeli wyników ---
print(df_results)

# --- 4. Wizualizacja wyników ---
plt.figure(figsize=(10, 5))
df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar")
plt.title("Porównanie klasyfikatorów")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
