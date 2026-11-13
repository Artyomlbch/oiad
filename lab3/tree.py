import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
CRITERIA = {"info_gain": "Information gain", "gain_ratio": "Gain ratio", "gini": "Gini index"}


def load(path):
    df = pd.read_csv(path, names=COLUMNS, skipinitialspace=True, comment="|")
    df["income"] = df["income"].str.rstrip(".") == ">50K"
    return df


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def gini(p):
    return 2 * p * (1 - p)


class DecisionTree:
    def __init__(self, criterion, max_depth):
        self.criterion = criterion
        self.max_depth = max_depth

    def score(self, n_l, p_l, n, p):
        w = n_l / n
        imp = gini if self.criterion == "gini" else entropy
        gain = imp(p / n) - w * imp(p_l / n_l) - (1 - w) * imp((p - p_l) / (n - n_l))
        if self.criterion == "gain_ratio":
            gain = gain / entropy(w)
        return gain

    def best_split(self, X, y):
        n, p = len(y), y.sum()
        best = None
        for col in X.columns:
            x = X[col].to_numpy()
            if col in self.numeric:
                order = np.argsort(x, kind="stable")
                xs = x[order]
                values = (xs[:-1] + xs[1:]) / 2
                n_l = np.arange(1, n)
                p_l = np.cumsum(y[order])[:-1]
                ok = xs[:-1] != xs[1:]
            else:
                values, inv = np.unique(x, return_inverse=True)
                n_l = np.bincount(inv)
                p_l = np.bincount(inv, weights=y)
                ok = n_l < n
            if not ok.any():
                continue
            scores = self.score(n_l[ok], p_l[ok], n, p)
            i = scores.argmax()
            if best is None or scores[i] > best[0]:
                best = (scores[i], col, values[ok][i])
        return best

    def goes_left(self, node, X):
        if node["col"] in self.numeric:
            return (X[node["col"]] <= node["value"]).to_numpy()
        return (X[node["col"]] == node["value"]).to_numpy()

    def build(self, X, y, depth):
        node = {"n": len(y), "p": y.mean()}
        if depth == self.max_depth or node["p"] in (0, 1):
            return node
        best = self.best_split(X, y)
        if best is None or best[0] <= 0:
            return node
        node["col"], node["value"] = best[1], best[2]
        left = self.goes_left(node, X)
        node["left"] = self.build(X[left], y[left], depth + 1)
        node["right"] = self.build(X[~left], y[~left], depth + 1)
        return node

    def fit(self, df):
        X, y = df.drop(columns="income"), df["income"].to_numpy(dtype=float)
        self.numeric = set(X.select_dtypes("number").columns)
        self.root = self.build(X, y, 0)
        return self

    def predict(self, df, node=None):
        node = node or self.root
        if "col" not in node:
            return np.full(len(df), node["p"] >= 0.5)
        left = self.goes_left(node, df)
        result = np.empty(len(df), dtype=bool)
        result[left] = self.predict(df[left], node["left"])
        result[~left] = self.predict(df[~left], node["right"])
        return result

    def draw(self, path, title):
        boxes, edges = [], []

        def place(node, depth, next_x):
            if "col" not in node:
                x = next_x
                label = (">50K" if node["p"] >= 0.5 else "<=50K") + f"\nn={node['n']}"
            else:
                xl, next_x = place(node["left"], depth + 1, next_x)
                xr, next_x = place(node["right"], depth + 1, next_x)
                x = (xl + xr) / 2
                sign = "<=" if node["col"] in self.numeric else "="
                value = f"{node['value']:g}" if node["col"] in self.numeric else node["value"]
                label = f"{node['col']} {sign} {value}\nn={node['n']}, >50K {node['p']:.0%}"
                edges.extend([(x, -depth, xl, "да"), (x, -depth, xr, "нет")])
            boxes.append((x, -depth, label, "col" not in node))
            return x, next_x + ("col" not in node)

        leaves = place(self.root, 0, 0)[1]
        plt.figure(figsize=(max(8, leaves * 1.15), 1.5 + self.max_depth * 1.2))
        for x, y, xc, text in edges:
            plt.plot([x, xc], [y, y - 1], color="gray", zorder=1)
            plt.text((x + xc) / 2, y - 0.5, text, ha="center", fontsize=9, color="gray")
        for x, y, text, leaf in boxes:
            plt.text(x, y, text, ha="center", va="center", fontsize=9,
                     bbox=dict(boxstyle="round", fc="#cfe8cf" if leaf else "#dbe7f5"))
        plt.title(title)
        plt.axis("off")
        plt.savefig(path, dpi=150, bbox_inches="tight")


def evaluate(df, criterion, max_depth, train_size):
    train, test = train_test_split(df, train_size=train_size, random_state=0)
    tree = DecisionTree(criterion, max_depth).fit(train)
    y, pred = test["income"], tree.predict(test)
    return [accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred), f1_score(y, pred)]


if __name__ == "__main__":
    MAX_DEPTH = 3
    CRITERION = "gini"
    data, test = load("data.csv"), load("test.csv")

    for criterion in CRITERIA:
        tree = DecisionTree(criterion, MAX_DEPTH).fit(data)
        tree.draw(f"tree_{criterion}.png", f"Дерево решений, критерий {CRITERIA[criterion]}")
        print(f"{criterion}: accuracy на test.csv = {accuracy_score(test['income'], tree.predict(test)):.4f}")

    full = pd.concat([data, test], ignore_index=True)
    sizes = [0.6, 0.7, 0.8, 0.9]
    results = [evaluate(full, CRITERION, MAX_DEPTH, s) for s in sizes]
    for s, r in zip(sizes, results):
        print(f"{s:.0%}: accuracy {r[0]:.4f}, precision {r[1]:.4f}, recall {r[2]:.4f}, F {r[3]:.4f}")

    labels = [f"{s * 100:g}:{100 - s * 100:g}" for s in sizes]
    plt.figure()
    for i, name in enumerate(["accuracy", "precision", "recall", "F-мера"]):
        plt.plot(labels, [r[i] for r in results], marker="o", label=name)
    plt.title(f"Качество классификации (критерий {CRITERIA[CRITERION]})")
    plt.xlabel("Обучающая : тестовая выборка, %")
    plt.ylabel("Значение показателя")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("metrics.png", dpi=150, bbox_inches="tight")

    plt.show()
