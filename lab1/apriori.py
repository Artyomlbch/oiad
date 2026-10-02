import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


class Apriori:
    def __init__(self, path):
        self.checks = self.read_checks(path)
        self.count = len(self.checks)
        self.start = self.get_start()
        self.all_frequent_items = {}

    def read_checks(self, path):
        for enc in ("utf-8-sig", "cp1251"):
            try:
                checks = []
                with open(path, encoding=enc) as f:
                    for line in f:
                        check = set(p.strip() for p in line.split(",") if p.strip())
                        if check:
                            checks.append(check)
                return checks
            except UnicodeDecodeError:
                continue
        raise ValueError("Не удалось прочитать файл " + path)

    def supp(self, products):
        products = set(products)
        support = 0
        for check in self.checks:
            if products <= check:
                support += 1
        return support / self.count

    # получение список всех продуктов без повторения (кандидаты длиной 1)
    def get_start(self):
        start = set()
        for check in self.checks:
            start |= check
        return sorted(start)

    def remove(self, min_supp, itemsets: dict):
        return {prod: supp for prod, supp in itemsets.items() if supp >= min_supp}

    # первый шаг, подсчет одиночных товаров
    def first_step(self, min_supp):
        current_dict = {}
        for product in self.start:
            current_dict[(product,)] = self.supp([product])
        return self.remove(min_supp, current_dict)

    def get_new_itemsets(self, itemsets):
        itemsets_set = set(itemsets)
        products = sorted(set(p for itemset in itemsets for p in itemset))
        new_itemsets = set()
        for itemset in itemsets:
            for product in products:
                if product in itemset:
                    continue
                candidate = tuple(sorted(itemset + (product,)))
                if candidate in new_itemsets:
                    continue
                if all(sub in itemsets_set for sub in combinations(candidate, len(itemset))):
                    new_itemsets.add(candidate)
        return sorted(new_itemsets)

    def algorithm(self, min_supp):
        self.all_frequent_items = {}
        frequent = self.first_step(min_supp)

        while frequent:
            self.all_frequent_items.update(frequent)
            # получаем кандидатов длиной k+1
            candidates = self.get_new_itemsets(list(frequent.keys()))
            candidates_dict = {}
            for cand in candidates:
                candidates_dict[cand] = self.supp(cand)
            frequent = self.remove(min_supp, candidates_dict)

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

        lines = [f"Чеков: {self.count}, min_supp = {min_supp}, "
                 f"порядок: {order}, найдено наборов: {len(result)}"]
        for items, supp in result:
            lines.append(f"{supp:.4f}  {{{', '.join(items)}}}")

        print("\n".join(lines))


    def visual(self, min_supps):
        times = []
        counts = []   # для каждого порога: {длина набора: количество наборов}
        for sup in min_supps:
            start_time = time.perf_counter()
            self.algorithm(sup)
            times.append(time.perf_counter() - start_time)

            by_len = {}
            for item in self.all_frequent_items:
                by_len[len(item)] = by_len.get(len(item), 0) + 1
            counts.append(by_len)
            print(f"min_supp = {sup}: время {times[-1]:.3f} с, наборов по длинам {by_len}")

        x_labels = [f"{s * 100:g}%" for s in min_supps]

        # диаграмма: быстродействие
        plt.figure()
        bars = plt.bar(x_labels, times)
        plt.bar_label(bars, fmt="%.3f")
        plt.title("Быстродействие алгоритма при разных min_supp")
        plt.xlabel("Порог min_supp")
        plt.ylabel("Время (секунды)")
        plt.savefig("time.png", dpi=150, bbox_inches="tight")

        # диаграмма: количество наборов разной длины
        plt.figure()
        all_lens = sorted(set(l for f in counts for l in f))
        x = np.arange(len(x_labels))
        width = 0.8 / max(len(all_lens), 1)

        for i, l in enumerate(all_lens):
            y = [f.get(l, 0) for f in counts]
            bars = plt.bar(x + i * width, y, width, label=f"Длина {l}")
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