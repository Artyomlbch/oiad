import timeit
from itertools import combinations

import matplotlib.pyplot as plt

EPS = 1e-9


class Apriori:
    def __init__(self, path):
        self.checks = self.read_checks(path)
        self.count = len(self.checks)
        self.all_frequent_items = {}
        self.all_rules = []

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

    def rules(self, min_conf):
        self.all_rules = []
        for itemset, supp in self.all_frequent_items.items():
            for k in range(1, len(itemset)):
                for ante in combinations(itemset, k):
                    conf = supp / self.all_frequent_items[ante]
                    if conf >= min_conf - EPS:
                        cons = tuple(p for p in itemset if p not in ante)
                        self.all_rules.append((ante, cons, supp, conf))
        return self.all_rules

    def sort_rules(self, order):
        rules = list(self.all_rules)
        if order == "support":
            rules.sort(key=lambda r: (-r[2], -r[3], r[0], r[1]))
        elif order == "lex":
            rules.sort(key=lambda r: (r[0], r[1]))
        else:
            raise ValueError("order должен быть 'support' или 'lex'")
        return rules

    def print_rules(self, min_supp, min_conf, order):
        self.algorithm(min_supp)
        self.rules(min_conf)
        rules = self.sort_rules(order)
        print(f"Чеков: {self.count}, min_supp = {min_supp}, min_conf = {min_conf}, "
              f"порядок: {order}, найдено правил: {len(rules)}")
        print("supp    conf    правило")
        for ante, cons, supp, conf in rules:
            print(f"{supp:.4f}  {conf:.4f}  {{{', '.join(ante)}}} → {{{', '.join(cons)}}}")

    def visual_rules(self, min_supp, min_confs):
        self.algorithm(min_supp)
        times = []
        counts = []
        for conf in min_confs:
            times.append(timeit.timeit(lambda: self.rules(conf), number=100) / 100 * 1000)
            counts.append(len(self.all_rules))
            print(f"min_conf = {conf}: время {times[-1]:.3f} мс, правил {counts[-1]}")

        x_labels = [f"{c * 100:g}%" for c in min_confs]

        plt.figure()
        bars = plt.bar(x_labels, times)
        plt.bar_label(bars, fmt="%.3f")
        plt.title(f"Быстродействие поиска правил (min_supp = {min_supp * 100:g}%)")
        plt.xlabel("Порог min_conf")
        plt.ylabel("Время (миллисекунды)")
        plt.savefig("rules_time.png", dpi=150, bbox_inches="tight")

        plt.figure()
        bars = plt.bar(x_labels, counts, color="tab:orange")
        plt.bar_label(bars)
        plt.title(f"Количество найденных правил (min_supp = {min_supp * 100:g}%)")
        plt.xlabel("Порог min_conf")
        plt.ylabel("Количество правил")
        plt.savefig("rules_count.png", dpi=150, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    PATH = "baskets.csv"
    MIN_SUPP = 0.01
    MIN_CONF = 0.4
    ORDER = "support"

    a = Apriori(PATH)
    a.print_rules(MIN_SUPP, MIN_CONF, ORDER)

    a.visual_rules(MIN_SUPP, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
