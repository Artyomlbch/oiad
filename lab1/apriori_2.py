import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

# допуск при сравнении дробных чисел: 7/10 может посчитаться как 0.69999999...
EPS = 1e-9
# стрелка в записи правила (если консоль ругается на символ — замените на "->")
ARROW = "→"


class Apriori:
    def __init__(self, path):
        self.checks = self.read_checks(path)
        self.count = len(self.checks)
        self.start = self.get_start()
        self.all_frequent_items = {}
        # список правил: (антецедент, консеквент, поддержка, достоверность)
        self.all_rules = []

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

    # ===================== АССОЦИАТИВНЫЕ ПРАВИЛА (новое) =====================

    def rules(self, min_conf):
        """
        Строит правила A -> C из уже найденных частых наборов (all_frequent_items).

        Для частого набора X и его непустого собственного подмножества C (консеквента):
            A = X \\ C                       — антецедент
            supp(A -> C) = supp(X)
            conf(A -> C) = supp(X) / supp(A)

        supp(A) берём из all_frequent_items: подмножество частого набора тоже частое,
        поэтому повторно проходить по чекам не нужно.
        """
        self.all_rules = []
        for itemset, supp_x in self.all_frequent_items.items():
            if len(itemset) < 2:
                continue  # из одного товара правило не построить

            # начинаем с консеквентов из одного товара
            consequents = [(p,) for p in itemset]
            while consequents and len(consequents[0]) < len(itemset):
                passed = []
                for cons in consequents:
                    ante = tuple(p for p in itemset if p not in cons)
                    conf = supp_x / self.all_frequent_items[ante]
                    if conf >= min_conf - EPS:
                        passed.append(cons)
                        self.all_rules.append((ante, cons, supp_x, conf))

                # Отсечение: если A -> C не прошло порог, то правило с более широким
                # консеквентом (C' ⊃ C) тоже не пройдёт — у него антецедент меньше,
                # поддержка антецедента больше, значит достоверность меньше.
                # Поэтому расширяем только прошедшие консеквенты — так же,
                # как в Apriori расширяются только частые наборы.
                consequents = self.get_new_itemsets(passed)

        return self.all_rules

    def sort_rules(self, order, rules=None):
        rules = list(self.all_rules if rules is None else rules)
        if order == "support":
            # по убыванию поддержки, при равной поддержке — по убыванию достоверности
            rules.sort(key=lambda r: (-r[2], -r[3], r[0], r[1]))
        elif order == "lex":
            # лексикографически: сначала по антецеденту, потом по консеквенту
            rules.sort(key=lambda r: (r[0], r[1]))
        else:
            raise ValueError("order должен быть 'support' или 'lex'")
        return rules

    def format_rule(self, rule):
        ante, cons, supp, conf = rule
        # lift = conf / supp(C): во сколько раз A повышает вероятность C
        # (не обязателен по заданию, но помогает в анализе: lift > 1 — реальная связь)
        lift = conf / self.all_frequent_items[cons]
        return (f"{supp:.4f}  {conf:.4f}  {lift:5.2f}  "
                f"{{{', '.join(ante)}}} {ARROW} {{{', '.join(cons)}}}")

    def print_rules(self, min_supp, min_conf, order, max_items=None):
        self.algorithm(min_supp)
        self.rules(min_conf)

        rules = self.all_rules
        if max_items is not None:
            # антецедент и консеквент вместе содержат не более max_items объектов
            rules = [r for r in rules if len(r[0]) + len(r[1]) <= max_items]
        rules = self.sort_rules(order, rules)

        header = (f"Чеков: {self.count}, min_supp = {min_supp}, min_conf = {min_conf}, "
                  f"порядок: {order}, найдено правил: {len(rules)}")
        if max_items is not None:
            header += f" (не более {max_items} объектов в правиле)"

        lines = [header, f"{'supp':<6}  {'conf':<6}  {'lift':>5}  правило"]
        for rule in rules:
            lines.append(self.format_rule(rule))
        text = "\n".join(lines)

        print(text + "\n")

        return rules

    def visual_rules(self, min_supp, min_confs, repeats=5):
        # При фиксированной поддержке частые наборы одни и те же,
        # поэтому ищем их один раз, а замеряем только построение правил.
        start_time = time.perf_counter()
        self.algorithm(min_supp)
        print(f"min_supp = {min_supp}: частых наборов {len(self.all_frequent_items)}, "
              f"поиск наборов занял {time.perf_counter() - start_time:.3f} с")

        times = []   # среднее время построения правил, мс
        counts = []  # количество правил
        for conf in min_confs:
            total = 0.0
            for _ in range(repeats):  # повторяем, чтобы сгладить случайные колебания
                start_time = time.perf_counter()
                self.rules(conf)
                total += time.perf_counter() - start_time
            times.append(total / repeats * 1000)
            counts.append(len(self.all_rules))
            print(f"min_conf = {conf}: время {times[-1]:.3f} мс, правил {counts[-1]}")

        x_labels = [f"{c * 100:g}%" for c in min_confs]

        # диаграмма: быстродействие поиска правил
        plt.figure()
        bars = plt.bar(x_labels, times)
        plt.bar_label(bars, fmt="%.3f")
        plt.title(f"Быстродействие поиска правил (min_supp = {min_supp * 100:g}%)")
        plt.xlabel("Порог min_conf")
        plt.ylabel("Время (миллисекунды)")
        plt.savefig("rules_time.png", dpi=150, bbox_inches="tight")

        # диаграмма: количество найденных правил
        plt.figure()
        bars = plt.bar(x_labels, counts, color="tab:orange")
        plt.bar_label(bars)
        plt.title(f"Количество найденных правил (min_supp = {min_supp * 100:g}%)")
        plt.xlabel("Порог min_conf")
        plt.ylabel("Количество правил")
        plt.gca().yaxis.get_major_locator().set_params(integer=True)  # только целые деления
        plt.savefig("rules_count.png", dpi=150, bbox_inches="tight")

        plt.show()

    # ====================================================================

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
    MIN_SUPP = 0.01    # фиксированный порог поддержки
    MIN_CONF = 0.50    # порог достоверности
    ORDER = "support"  # "support" — по убыванию поддержки, "lex" — лексикографически
    MAX_ITEMS = 7      # для п.4: не более 7 объектов в правиле

    a = Apriori(PATH)

    # п.1: все правила в виде антецедент → консеквент с поддержкой и достоверностью
    a.print_rules(MIN_SUPP, MIN_CONF, ORDER)

    # п.4: правила, где антецедент + консеквент содержат не более 7 объектов
    a.print_rules(MIN_SUPP, MIN_CONF, ORDER, max_items=MAX_ITEMS)

    # п.2–3: min_supp фиксирован, min_conf от 70% до 95% с шагом 5%
    min_confs = [round(MIN_CONF + 0.05 * i, 2) for i in range(6)]
    a.visual_rules(MIN_SUPP, min_confs)