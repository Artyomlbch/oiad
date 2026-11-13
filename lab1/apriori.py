import time
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


class Apriori:
    def __init__(self, path):
        self.checks = self.read_checks(path)
        self.count = len(self.checks)
        self.all_frequent_items = {}

    def read_checks(self, path):
        for enc in ("utf-8-sig", "cp1251"):
            try:
                with open(path, encoding=enc) as f:
                    checks = [{p.strip() for p in line.split(",")} - {""} for line in f]
                return [c for c in checks if c]
            except UnicodeDecodeError:
                pass
        raise ValueError("Не удалось прочитать файл " + path)

    def supp(self, products):
        products = set(products)
        return sum(products <= check for check in self.checks) / self.count

    def remove(self, min_supp, candidates):
        supports = {c: self.supp(c) for c in candidates}
        return {c: s for c, s in supports.items() if s >= min_supp}

    def get_new_itemsets(self, itemsets):
        itemsets_set = set(itemsets)
        products = sorted({p for itemset in itemsets for p in itemset})
        new_itemsets = set()
        for itemset in itemsets:
            for product in products:
                if product in itemset:
                    continue
                candidate = tuple(sorted(itemset + (product,)))
                if all(sub in itemsets_set for sub in combinations(candidate, len(itemset))):
                    new_itemsets.add(candidate)
        return sorted(new_itemsets)

    def algorithm(self, min_supp):
        self.all_frequent_items = {}
        products = sorted(set().union(*self.checks))
        frequent = self.remove(min_supp, [(p,) for p in products])
        while frequent:
            self.all_frequent_items.update(frequent)
            frequent = self.remove(min_supp, self.get_new_itemsets(list(frequent)))
        return self.all_frequent_items

    def sort_result(self, order):
        items = list(self.all_frequent_items.items())
        if order == "support":
            items.sort(key=lambda x: (-x[1], x[0]))
        elif order == "lex":
            items.sort(key=lambda x: x[0])
        else:
            raise ValueError("order должен быть 'support' или 'lex'")
        return items

    def print_result(self, min_supp, order):
        self.algorithm(min_supp)
        result = self.sort_result(order)
        print(f"Чеков: {self.count}, min_supp = {min_supp}, "
              f"порядок: {order}, найдено наборов: {len(result)}")
        for items, supp in result:
            print(f"{supp:.4f}  {{{', '.join(items)}}}")

    def visual(self, min_supps):
        times = []
        counts = []
        for sup in min_supps:
            start_time = time.perf_counter()
            self.algorithm(sup)
            times.append(time.perf_counter() - start_time)
            counts.append(Counter(len(item) for item in self.all_frequent_items))
            print(f"min_supp = {sup}: время {times[-1]:.3f} с, наборов по длинам {dict(counts[-1])}")

        x_labels = [f"{s * 100:g}%" for s in min_supps]

        plt.figure()
        bars = plt.bar(x_labels, times)
        plt.bar_label(bars, fmt="%.3f")
        plt.title("Быстродействие алгоритма при разных min_supp")
        plt.xlabel("Порог min_supp")
        plt.ylabel("Время (секунды)")
        plt.savefig("time.png", dpi=150, bbox_inches="tight")

        plt.figure()
        all_lens = sorted(set().union(*counts))
        x = np.arange(len(x_labels))
        width = 0.8 / len(all_lens)
        for i, l in enumerate(all_lens):
            bars = plt.bar(x + i * width, [c[l] for c in counts], width, label=f"Длина {l}")
            plt.bar_label(bars, fontsize=8)
        plt.xticks(x + width * (len(all_lens) - 1) / 2, x_labels)
        plt.title("Количество частых наборов разной длины")
        plt.xlabel("Порог min_supp")
        plt.ylabel("Количество наборов")
        plt.legend()
        plt.savefig("counts.png", dpi=150, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    PATH = "baskets.csv"
    MIN_SUPP = 0.15
    ORDER = "support"

    a = Apriori(PATH)
    a.print_result(MIN_SUPP, ORDER)

    a.visual([0.01, 0.03, 0.05, 0.1, 0.15])
