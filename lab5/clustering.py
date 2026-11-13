import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load(path):
    df = pd.read_csv(path).drop(columns=["Row", "CustomerId"])
    return df.fillna(df.mean()).astype(float)


def add_noise(df, fraction, seed=0):
    rng = np.random.default_rng(seed)
    noisy = df.copy()
    rows = rng.choice(len(df), round(fraction * len(df)), replace=False)
    noisy.iloc[rows] += rng.uniform(-3, 3, (len(rows), df.shape[1])) * df.std().to_numpy()
    return noisy


def scale(df):
    return ((df - df.mean()) / df.std()).to_numpy()


def k_means(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    return model.labels_, model.cluster_centers_, model.inertia_


def k_medoids(X, k, n_init=10, seed=0):
    rng = np.random.default_rng(seed)
    D = cdist(X, X)
    best = None
    for _ in range(n_init):
        medoids = rng.choice(len(X), k, replace=False)
        for _ in range(100):
            labels = D[:, medoids].argmin(axis=1)
            new = np.array([np.flatnonzero(labels == j)[D[np.ix_(labels == j, labels == j)].sum(axis=1).argmin()]
                            for j in range(k)])
            if (new == medoids).all():
                break
            medoids = new
        cost = D[np.arange(len(X)), medoids[labels]].sum()
        if best is None or cost < best[0]:
            best = (cost, labels, medoids)
    return best[1], X[best[2]], best[0]


def scatter(ax, X, labels, centers, title):
    pca = PCA(2).fit(X)
    points, c = pca.transform(X), pca.transform(centers)
    ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=10)
    ax.scatter(c[:, 0], c[:, 1], c="black", marker="x", s=80)
    ax.set_title(title)
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")


ALGORITHMS = {"k-Means": k_means, "k-Medoids": k_medoids}
ERRORS = {"k-Means": "Сумма квадратов расстояний до центров", "k-Medoids": "Сумма расстояний до медоидов"}

if __name__ == "__main__":
    PATH = "customers.csv"
    K = 3
    K_VALUES = list(range(3, 10))
    NOISE = [0.01, 0.03, 0.05, 0.1]

    df = load(PATH)
    X = scale(df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (name, algorithm) in zip(axes, ALGORITHMS.items()):
        labels, centers, err = algorithm(X, K)
        print(f"{name}, k = {K}, ошибка = {err:.1f}")
        print(df.assign(cluster=labels))
        scatter(ax, X, labels, centers, f"{name}, k = {K}")
    fig.tight_layout()
    plt.savefig("clusters.png", dpi=150, bbox_inches="tight")

    fig_err, axes_err = plt.subplots(1, 2, figsize=(10, 4.2))
    fig_cl, axes_cl = plt.subplots(len(NOISE), 2, figsize=(10, 3.6 * len(NOISE)))
    for row, (name, algorithm) in enumerate(ALGORITHMS.items()):
        for col, fraction in enumerate([0] + NOISE):
            Xn = scale(add_noise(df, fraction))
            errors = [algorithm(Xn, k)[2] for k in K_VALUES]
            print(f"{name}, шум {fraction:.0%}, ошибка при k = {K_VALUES[0]}..{K_VALUES[-1]}:", ", ".join(f"{e:.1f}" for e in errors))
            axes_err[row].plot(K_VALUES, errors, marker="o", label=f"шум {fraction:.0%}")
            if fraction:
                labels, centers, _ = algorithm(Xn, K)
                scatter(axes_cl[col - 1, row], Xn, labels, centers, f"{name}, шум {fraction:.0%}, k = {K}")
        axes_err[row].set_title(name)
        axes_err[row].set_xlabel("Число кластеров k")
        axes_err[row].set_ylabel(ERRORS[name])
        axes_err[row].legend()
        axes_err[row].grid(alpha=0.3)
    fig_err.tight_layout()
    fig_err.savefig("error.png", dpi=150, bbox_inches="tight")
    fig_cl.tight_layout()
    fig_cl.savefig("noise_clusters.png", dpi=150, bbox_inches="tight")

    plt.show()
