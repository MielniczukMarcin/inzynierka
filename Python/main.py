import matplotlib.pyplot as plt
from load_data import load_dataset
from train_models import train_and_evaluate
from evaluate import compare_results

# --- 1. Wybór zbioru danych ---
dataset_name = input("Wybierz zbiór danych (iris, wine, breast_cancer): ").strip().lower()
X_train, X_test, y_train, y_test = load_dataset(dataset_name)

# --- 2. Trenowanie i ewaluacja modeli ---
df_results = train_and_evaluate(X_train, X_test, y_train, y_test)

# --- 3. Porównanie wyników ---
compare_results(df_results, save_to_csv=True, filename=f"results_{dataset_name}.csv")

# --- 4. Wizualizacja wyników ---
plt.figure(figsize=(10, 5))
df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(kind="bar")
plt.title(f"Porównanie klasyfikatorów - {dataset_name}")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
