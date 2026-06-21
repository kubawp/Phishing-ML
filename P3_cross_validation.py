from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


dane = fetch_ucirepo(id=967)
X = dane.data.features
y = dane.data.targets.iloc[:, 0]
X = X.select_dtypes(include=["number"])
X = X.drop(columns=["URLSimilarityIndex"])


modele = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian NB": GaussianNB()
}


print(f"\n{'Model':<20} {'Średnia':>10} {'Odchylenie':>12} {'Min':>8} {'Max':>8}")
print("-" * 62)

for nazwa, model in modele.items():
    wyniki = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    srednia = wyniki.mean()
    odchylenie = wyniki.std()
    minimum = wyniki.min()
    maksimum = wyniki.max()

    print(f"{nazwa:<20} {srednia:>10.4f} {odchylenie:>12.4f} {minimum:>8.4f} {maksimum:>8.4f}")