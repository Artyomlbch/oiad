import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


class Apriori:
    def __init__(self, path):
        self.checks = self.read_checks(path)   # список чеков, каждый чек - множество товаров
        self.count = len(self.checks)          # сколько всего чеков
        self.start = self.get_start()          # все различные товары
        self.all_frequent_items = {}           # результат: {набор: поддержка}

    def read_checks(self, path):
        # пробуем utf-8, если не получилось - cp1251 (виндовая кодировка для русского)
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
        # поддержка = доля чеков, в которых есть ВСЕ товары набора
        products = set(products)
        support = 0
        for check in self.checks:
            if products <= check:
                support += 1
        return support / self.count

    def get_start(self):
        start = set()
        for check in self.checks:
            start |= check
        return sorted(start)

    def get_new_links(self, links):
        # из частых наборов длины k строим кандидатов длины k+1:
        # к каждому частому набору добавляем по одному товару
        links_set = set(links)
        products = sorted(set(p for link in links for p in link))
        new_links = set()
        for link in links:
            for product in products:
                if product in link:
                    continue
                candidate = tuple(sorted(link + (product,)))
                if candidate in new_links:
                    continue
                # отсечение (главная идея Apriori): если хоть одно подмножество
                # длины k не частое, то и кандидат частым быть не может
                if all(sub in links_set for sub in combinations(candidate, len(link))):
                    new_links.add(candidate)
        return sorted(new_links)

    def remove(self, min_supp, links: dict):
        # оставляем только наборы с поддержкой не меньше порога
        return {prod: supp for prod, supp in links.items() if supp >= min_supp}

    def first_step(self, min_supp):
        # шаг 1: одиночные товары
        current_dict = {}
        for product in self.start:
            current_dict[(product,)] = self.supp([product])
        return self.remove(min_supp, current_dict)

    def algorithm(self, min_supp):
        self.all_frequent_items = {}
        frequent = self.first_step(min_supp)

        while frequent:
            self.all_frequent_items.update(frequent)
            candidates = self.get_new_links(list(frequent.keys()))
            candidates_dict = {}
            for cand in candidates:
                candidates_dict[cand] = self.supp(cand)
            frequent = self.remove(min_supp, candidates_dict)

        return self.all_frequent_items

    def sort_result(self, order):
        items = list(self.all_frequent_items.items())
        if order == "support":
            # по убыванию поддержки (при равной поддержке - по алфавиту)
            items.sort(key=lambda x: (-x[1], x[0]))
        elif order == "lex":
            # лексикографический порядок
            items.sort(key=lambda x: x[0])
        else:
            raise ValueError("order должен быть 'support' или 'lex'")
        return items

    def print_result(self, min_supp, order, out_path=None):
        self.algorithm(min_supp)
        result = self.sort_result(order)

        lines = [f"Чеков: {self.count}, min_supp = {min_supp}, "
                 f"порядок: {order}, найдено наборов: {len(result)}"]
        for items, supp in result:
            lines.append(f"{supp:.4f}  {{{', '.join(items)}}}")

        print("\n".join(lines))
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"Результат сохранён в {out_path}")

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

        # диаграмма 1: быстродействие
        plt.figure()
        bars = plt.bar(x_labels, times)
        plt.bar_label(bars, fmt="%.3f")
        plt.title("Быстродействие алгоритма при разных min_supp")
        plt.xlabel("Порог min_supp")
        plt.ylabel("Время (секунды)")
        plt.savefig("time.png", dpi=150, bbox_inches="tight")

        # диаграмма 2: количество наборов разной длины
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
    # ---------- параметры программы ----------
    PATH = "baskets.csv"   # набор данных
    MIN_SUPP = 0.03        # порог поддержки (0.03 = 3%)
    ORDER = "support"      # "support" - по убыванию поддержки, "lex" - лексикографически

    a = Apriori(PATH)
    a.print_result(MIN_SUPP, ORDER, out_path="result.txt")

    # ---------- эксперименты для отчёта ----------
    a.visual([0.01, 0.03, 0.05, 0.1, 0.15])