# Phishing-ML
Projekt dotyczy klasyfikacji adresów URL jako legalne lub phishingowe z użyciem prostych modeli uczenia maszynowego.

W projekcie wykorzystano dataset PhiUSIIL Phishing URL Dataset, który został pobrany za pomocą biblioteki `ucimlrepo`.

## Co robi projekt?

Projekt obejmuje:

- wczytanie i przygotowanie danych,
- sprawdzenie podstawowych informacji o zbiorze,
- wykonanie kilku wykresów i wizualizacji,
- trenowanie modeli klasyfikacyjnych,
- porównanie wyników modeli,
- walidację krzyżową,
- sprawdzenie wpływu rozmiaru zbioru treningowego na wynik.

## Użyte modele

W projekcie porównano:

- Decision Tree,
- KNN,
- Gaussian Naive Bayes.

## Technologie

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- ucimlrepo

## Uruchomienie

Najpierw należy zainstalować biblioteki:

```bash
pip install -r requirements.txt
