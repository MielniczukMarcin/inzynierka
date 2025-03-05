Porównanie klasyfikatorów w Pythonie

Ten projekt porównuje różne klasyfikatory uczenia maszynowego przy użyciu zbioru danych Iris.

Struktura projektu:

inzynierka/Python/

    main.py – Główne wywołanie programu
    load_data.py – Moduł do wczytywania danych
    train_models.py – Moduł do trenowania modeli
    evaluate.py – Moduł do obliczania metryk ewaluacyjnych
    README.md – Dokumentacja projektu
Rozbudowa

Aby dodać nowy zbiór danych:

    Edytuj load_data.py, aby dodać nową funkcję ładującą dane.
    W main.py zmień funkcję load_dataset() na nową.

Aby dodać nowy model:

    W train_models.py dodaj nowy klasyfikator do słownika classifiers.

Autor

Projekt inżynierski – Marcin (mm51621)
