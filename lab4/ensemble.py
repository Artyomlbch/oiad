import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
MODELS = {"bagging": BaggingClassifier, "random_forest": RandomForestClassifier, "boosting": AdaBoostClassifier}
NAMES = {"bagging": "бэггинг", "random_forest": "случайный лес", "boosting": "бустинг"}
METRICS = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score, "F-мера": f1_score}


def load(path):
    df = pd.read_csv(path, names=COLUMNS, skipinitialspace=True, comment="|")
    df["income"] = df["income"].str.rstrip(".") == ">50K"
    return df


def evaluate(model, X_train, y_train, X_test, y_test):
    pred = model.fit(X_train, y_train).predict(X_test)
    return [metric(y_test, pred) for metric in METRICS.values()]


if __name__ == "__main__":
    TECHNIQUE = "random_forest"
    PARAMS = {"bagging": {"max_samples": 0.5}, "random_forest": {"max_depth": 15}, "boosting": {"learning_rate": 1.0}}
    SIZES = list(range(50, 101, 10))

    data, test = load("data.csv"), load("test.csv")
    X = pd.get_dummies(pd.concat([data, test]).drop(columns="income"))
    X_train, X_test = X[:len(data)], X[len(data):]
    y_train, y_test = data["income"], test["income"]

    tree = evaluate(DecisionTreeClassifier(max_depth=3, random_state=0), X_train, y_train, X_test, y_test)
    print("дерево из ПР3:", ", ".join(f"{name} {v:.4f}" for name, v in zip(METRICS, tree)))

    results = []
    for n in SIZES:
        model = MODELS[TECHNIQUE](n_estimators=n, random_state=0, **PARAMS[TECHNIQUE])
        results.append(evaluate(model, X_train, y_train, X_test, y_test))
        print(f"{n} участников:", ", ".join(f"{name} {v:.4f}" for name, v in zip(METRICS, results[-1])))

    plt.figure()
    for i, name in enumerate(METRICS):
        line, = plt.plot(SIZES, [r[i] for r in results], marker="o", label=name)
        plt.axhline(tree[i], color=line.get_color(), linestyle="--", alpha=0.6)
    plt.plot([], [], color="gray", linestyle="--", label="дерево из ПР3")
    plt.title(f"Качество классификации, {NAMES[TECHNIQUE]}")
    plt.xlabel("Количество участников ансамбля")
    plt.ylabel("Значение показателя")
    plt.xticks(SIZES)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(alpha=0.3)
    plt.savefig(f"{TECHNIQUE}.png", dpi=150, bbox_inches="tight")
    plt.show()
