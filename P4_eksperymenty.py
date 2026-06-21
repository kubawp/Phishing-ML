import matplotlib
matplotlib.use("TkAgg")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

OUTPUT_DIR = "wyniki/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

url_features = [
    "URLLength", "DegitRatioInURL", "LetterRatioInURL",
    "NoOfDegitsInURL", "NoOfLettersInURL",
    "SpacialCharRatioInURL", "NoOfOtherSpecialCharsInURL"
]

domain_features = [
    "DomainLength", "IsDomainIP", "IsHTTPS",
    "NoOfSubDomain", "TLDLength", "TLDLegitimateProb"
]

html_features = [
    "NoOfExternalRef", "NoOfSelfRef", "NoOfEmptyRef",
    "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfPopup", "NoOfiFrame"
]


print("Pobieranie datasetu...")

dataset = fetch_ucirepo(id=967)

X = dataset.data.features.select_dtypes(include=[np.number]).fillna(0)
y = dataset.data.targets.values.ravel()

X_values = X.values

modele = {
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(max_depth=10)
}

metryki = {
    "Accuracy": accuracy_score,
    "F1": f1_score,
    "Precision": precision_score,
    "Recall": recall_score
}

cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

# eksperyment 1 - porównanie klasyfikatorów


print("\n=== Eksperyment 1: Porownanie klasyfikatorow ===")
print("(Odniesienie: Table 3, Prasad & Chandra 2024)")

exp1 = {}

for nazwa in modele:
    exp1[nazwa] = {}

    for metryka in metryki:
        exp1[nazwa][metryka] = []


for i, (train, test) in enumerate(cv.split(X_values, y)):
    for nazwa, model in modele.items():
        model.fit(X_values[train], y[train])
        pred = model.predict(X_values[test])

        for nazwa_metryki, funkcja in metryki.items():
            wynik = funkcja(y[test], pred)
            exp1[nazwa][nazwa_metryki].append(wynik)

    print("Fold %d/10" % (i + 1))


print("\n%-6s %-10s %-8s %-10s %-8s" % ("Model", "Accuracy", "F1", "Precision", "Recall"))
print("-" * 46)

for nazwa in modele:
    for metryka in metryki:
        exp1[nazwa][metryka] = np.array(exp1[nazwa][metryka])

    print("%-6s %.4f     %.4f   %.4f     %.4f" % (
        nazwa,
        exp1[nazwa]["Accuracy"].mean(),
        exp1[nazwa]["F1"].mean(),
        exp1[nazwa]["Precision"].mean(),
        exp1[nazwa]["Recall"].mean()
    ))

# eksperyment 2 - wpływ rozmiaru zbioru treningowego


print("\n=== Eksperyment 2: Wplyw rozmiaru zbioru treningowego ===")
print("(Odniesienie: Table 7, Prasad & Chandra 2024)")

train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
exp2 = {nazwa: [] for nazwa in modele}

for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X_values,
        y,
        train_size=size,
        stratify=y
    )

    for nazwa, model in modele.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        exp2[nazwa].append(acc)

    print("Train size %.0f%% zakonczone" % (size * 100))


print("\n%-6s " % "Model" + " ".join(["%.0f%%" % (s * 100) for s in train_sizes]))
print("-" * 70)

for nazwa in modele:
    row = "%-6s " % nazwa
    row += " ".join(["%.4f" % wynik for wynik in exp2[nazwa]])
    print(row)

# eksperyment 3 - porównanie grup cech


print("\n=== Eksperyment 3: Porownanie grup cech (KNN) ===")
print("(Odniesienie: sekcja 3.1 - URL, HTML i derived features)")

grupy = {
    "URL features": url_features,
    "Domain features": domain_features,
    "HTML features": html_features
}

exp3 = {}

for nazwa_grupy, kolumny in grupy.items():
    kolumny = [c for c in kolumny if c in X.columns]

    X_grupa = X[kolumny].values
    wyniki = []

    for train, test in cv.split(X_grupa, y):
        model = KNeighborsClassifier()
        model.fit(X_grupa[train], y[train])

        pred = model.predict(X_grupa[test])
        acc = accuracy_score(y[test], pred)

        wyniki.append(acc)

    exp3[nazwa_grupy] = np.array(wyniki)

    print("%s | mean=%.4f | std=%.4f" % (
        nazwa_grupy,
        exp3[nazwa_grupy].mean(),
        exp3[nazwa_grupy].std()
    ))

# wykres 1 - porównanie klasyfikatorów


metric_names = list(metryki.keys())
model_names = list(modele.keys())

x = np.arange(len(model_names))
width = 0.2
colors = ["#3498db", "#e67e22", "#9b59b6", "#e74c3c"]

fig, ax = plt.subplots(figsize=(12, 6))

for i, (metryka, kolor) in enumerate(zip(metric_names, colors)):
    wartosci = [exp1[nazwa][metryka].mean() for nazwa in model_names]

    ax.bar(
        x + (i - 1.5) * width,
        wartosci,
        width,
        label=metryka,
        color=kolor,
        edgecolor="black"
    )

ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0.95, 1.01)
ax.set_title("Eksperyment 1: Porownanie klasyfikatorow\n(ref. Table 3, Prasad & Chandra 2024)")
ax.set_ylabel("Wartosc metryki")
ax.legend()
ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "exp1_klasyfikatory.png", dpi=150)
plt.show()

# wykres 2 - wpływ rozmiaru zbioru treningowego


plt.figure(figsize=(10, 6))

for nazwa in model_names:
    plt.plot(
        [int(s * 100) for s in train_sizes],
        exp2[nazwa],
        marker="o",
        label=nazwa
    )

plt.xlabel("Rozmiar zbioru treningowego (%)")
plt.ylabel("Accuracy")
plt.title("Eksperyment 2: Wplyw rozmiaru zbioru treningowego\n(ref. Table 7, Prasad & Chandra 2024)")
plt.legend()
plt.grid(alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "exp2_krzywa_uczenia.png", dpi=150)
plt.show()

# wykres 3 - porównanie grup cech


group_names = list(exp3.keys())
means = [exp3[g].mean() for g in group_names]
stds = [exp3[g].std() for g in group_names]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    group_names,
    means,
    yerr=stds,
    capsize=6,
    color=["#3498db", "#e67e22", "#9b59b6"],
    edgecolor="black"
)

for bar, wynik in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        "%.4f" % wynik,
        ha="center"
    )

plt.title("Eksperyment 3: Porownanie grup cech (KNN)\n(ref. sekcja 3.1, Prasad & Chandra 2024)")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.05)
plt.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "exp3_grupy_cech.png", dpi=150)
plt.show()

with open(OUTPUT_DIR + "p4_results.pkl", "wb") as f:
    pickle.dump({
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3
    }, f)

print("\nWyniki zapisane do p4_results.pkl")