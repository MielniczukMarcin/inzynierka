import matplotlib.pyplot as plt
from load_data import load_dataset
from train_models import train_and_evaluate
from evaluate import compare_results  # Nowa funkcja do porównywania wyników

# --- 1. Wczytanie danych ---
X_train, X_test, y_train, y_test = load_dataset()

# --- 2. Trenowanie i ewaluacja modeli ---
df_results = train_and_evaluate(X_train, X_test, y_train, y_test)

# --- 3. Porównanie wyników ---
compare_results(df_results, save_to_csv=True)  # Automatycznie zapisuje wyniki do CSV

# --- 4. Wizualizacja wyników ---
plt.figure(figsize=(10, 5))
df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar")
plt.title("Porównanie klasyfikatorów")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
