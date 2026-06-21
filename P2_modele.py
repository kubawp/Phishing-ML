from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


dane = fetch_ucirepo(id=967)
X = dane.data.features
y = dane.data.targets.iloc[:, 0]
X = X.select_dtypes(include=["number"])
X = X.drop(columns=["URLSimilarityIndex"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

modele = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian NB": GaussianNB()
}


print(f"\n{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 60)

for nazwa, model in modele.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{nazwa:<20} {acc:>10.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")


print("\nMacierze pomyłek (wiersze=rzeczywiste, kolumny=przewidziane):")
print("             [Phishing  Legit]")

for nazwa, model in modele.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{nazwa}:")
    print(f"  Phishing  {cm[0]}")
    print(f"  Legit     {cm[1]}")


waznosci = modele["Decision Tree"].feature_importances_

tabela_cech = pd.DataFrame({
    "Cecha": X.columns,
    "Waznosc": waznosci
})

tabela_cech = tabela_cech.sort_values(by="Waznosc", ascending=False)

print("\nTop 5 najważniejszych cech dla Drzewa Decyzyjnego:")
print(tabela_cech.head(5))