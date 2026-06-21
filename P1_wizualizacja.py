import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

dane = fetch_ucirepo(id=967)
X = dane.data.features.select_dtypes(include=[np.number])
y = dane.data.targets.values.ravel()

print("Wymiary X:", X.shape)
print("Wymiary y:", y.shape)
print("Balans klas:")
print(pd.Series(y).value_counts())
print("Brakujące wartości:", X.isnull().sum().sum())


# wykres balansu klas
klasy = pd.Series(y).value_counts()

plt.figure()
klasy.plot(kind="bar", color=["#2ecc71", "#e74c3c"])
plt.title("Balans klas")
plt.xlabel("Klasa")
plt.ylabel("Liczba próbek")
plt.tight_layout()
plt.savefig("balans_klas.png", dpi=150)
plt.show()


# rozkład wybranych cech
cechy = ["URLLength", "DomainLength", "TLDLength"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

for i, cecha in enumerate(cechy):
    legit = X[y == 1][cecha]
    phishing = X[y == 0][cecha]

    if cecha == "URLLength":
        granica = X[cecha].quantile(0.99)
        legit = legit[legit <= granica]
        phishing = phishing[phishing <= granica]

    ax[i].hist(legit, bins=40, alpha=0.6, color="#2ecc71", label="Legit")
    ax[i].hist(phishing, bins=40, alpha=0.6, color="#e74c3c", label="Phishing")

    ax[i].set_title(cecha)
    ax[i].set_xlabel("Wartość")
    ax[i].set_ylabel("Liczba próbek")
    ax[i].legend()

plt.suptitle("Rozkład cech")
plt.tight_layout()
plt.savefig("rozklad_cech.png", dpi=150)
plt.show()


# korelacja cech z etykietą
df = X.copy()
df["label"] = y

korelacje = df.corr()["label"].abs().sort_values(ascending=False)

top15 = korelacje[1:16]

plt.figure(figsize=(10, 6))
plt.barh(top15.index[::-1], top15.values[::-1])
plt.title("Top 15 cech korelujących z etykietą")
plt.xlabel("Bezwzględna korelacja")
plt.tight_layout()
plt.savefig("korelacja_top15.png", dpi=150)
plt.show()


# macierz korelacji dla 10 najlepszych cech
top10 = korelacje.index[1:11]

plt.figure(figsize=(12, 10))
sns.heatmap(df[top10].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Macierz korelacji top 10 cech")
plt.tight_layout()
plt.savefig("macierz_korelacji.png", dpi=150)
plt.show()


# PCA 2D
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

losowe = np.random.choice(len(y), 5000, replace=False)

plt.figure()
plt.scatter(
    X_pca[losowe, 0],
    X_pca[losowe, 1],
    c=y[losowe],
    cmap="RdYlGn",
    alpha=0.4,
    s=5
)

wariancja = sum(pca.explained_variance_ratio_) * 100

plt.title("PCA 2D, wariancja: %.1f%%" % wariancja)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=150)
plt.show()