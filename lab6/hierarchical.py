import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

METHODS = {"single": "Single linkage", "complete": "Complete linkage", "average": "Group average", "ward": "Ward distance"}


def load(path):
    df = pd.read_csv(path).drop(columns=["Row", "CustomerId"])
    return df.fillna(df.mean()).astype(float)


def scale(df):
    return ((df - df.mean()) / df.std()).to_numpy()


def hierarchical(X, method, k):
    Z = linkage(X, method=method, metric="euclidean")
    return Z, fcluster(Z, k, criterion="maxclust") - 1


if __name__ == "__main__":
    PATH = "customers.csv"
    K = 3

    df = load(PATH)
    X = scale(df)
    points = PCA(2).fit_transform(X)
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
    print(f"k-Means: размеры кластеров {np.bincount(kmeans).tolist()}")

    fig_d, axes_d = plt.subplots(2, 2, figsize=(10, 8))
    fig_s, axes_s = plt.subplots(2, 2, figsize=(10, 8))
    for ax_d, ax_s, (method, name) in zip(axes_d.flat, axes_s.flat, METHODS.items()):
        Z, labels = hierarchical(X, method, K)
        print(f"{name}: размеры кластеров {np.bincount(labels).tolist()}, "
              f"сходство с k-Means (ARI) {adjusted_rand_score(kmeans, labels):.3f}")
        cut = (Z[-K, 2] + Z[-K + 1, 2]) / 2
        dendrogram(Z, ax=ax_d, truncate_mode="lastp", p=30, color_threshold=cut, leaf_font_size=7)
        ax_d.axhline(cut, color="gray", linestyle="--")
        ax_d.set_title(name)
        ax_s.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=8)
        ax_s.set_title(f"{name}, k = {K}")
    fig_d.supylabel("Расстояние между кластерами")
    fig_d.tight_layout()
    fig_d.savefig("dendrograms.png", dpi=150, bbox_inches="tight")
    fig_s.supxlabel("Главная компонента 1")
    fig_s.supylabel("Главная компонента 2")
    fig_s.tight_layout()
    fig_s.savefig("clusters.png", dpi=150, bbox_inches="tight")

    plt.show()
