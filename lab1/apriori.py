import csv
import math
import sys
from itertools import combinations

ENCODING = "cp1251"


def load_transactions(path):
    """Читает CSV: каждая строка -- одна корзина покупок.

    Возвращает список множеств: [{"молоко", "хлеб"}, {"яйца"}, ...]
    Повторы товара внутри корзины схлопываются: важен факт покупки,
    а не количество.
    """
    transactions = []
    with open(path, encoding=ENCODING, newline="") as f:
        for row in csv.reader(f):
            basket = set(cell.strip() for cell in row if cell.strip())
            if basket:
                transactions.append(basket)
    return transactions


def generate_candidates(prev_frequent, k):
    """Кандидаты длины k из частых наборов длины k-1.

    prev_frequent -- отсортированный список отсортированных кортежей.

    Соединение: два (k-1)-набора объединяются, если совпадают их первые
    k-2 элемента. Так каждый кандидат получается ровно один раз и сразу
    отсортированным:
        ("молоко", "сыр") + ("молоко", "хлеб") -> ("молоко", "сыр", "хлеб")

    Отсечение: если хотя бы одно (k-1)-подмножество кандидата не частое,
    то и кандидат частым быть не может -- выбрасываем, не считая поддержку.
    Это и есть главная идея Apriori.
    """
    prev_set = set(prev_frequent)
    candidates = []

    for i in range(len(prev_frequent)):
        a = prev_frequent[i]
        for j in range(i + 1, len(prev_frequent)):
            b = prev_frequent[j]
            if a[:k - 2] != b[:k - 2]:
                break                      # список отсортирован -- дальше не совпадёт
            candidate = a + (b[k - 2],)
            if all(sub in prev_set for sub in combinations(candidate, k - 1)):
                candidates.append(candidate)
    return candidates


def apriori(transactions, min_support):
    """Возвращает список пар (набор, поддержка), поддержка -- доля транзакций."""
    n = len(transactions)
    min_count = math.ceil(min_support * n)   # порог в штуках транзакций
    result = []

    # --- уровень 1: одиночные товары ---
    counts = {}
    for basket in transactions:
        for item in basket:
            counts[item] = counts.get(item, 0) + 1

    frequent = sorted((item,) for item, c in counts.items() if c >= min_count)
    for itemset in frequent:
        result.append((itemset, counts[itemset[0]] / n))

    # --- уровни 2, 3, ... пока находятся частые наборы ---
    k = 2
    while frequent:
        candidates = set(generate_candidates(frequent, k))
        if not candidates:
            break

        # один проход по базе: считаем поддержку всех кандидатов сразу
        counts = {}
        for basket in transactions:
            if len(basket) < k:
                continue
            for combo in combinations(sorted(basket), k):
                if combo in candidates:
                    counts[combo] = counts.get(combo, 0) + 1

        frequent = sorted(c for c, cnt in counts.items() if cnt >= min_count)
        for itemset in frequent:
            result.append((itemset, counts[itemset] / n))
        k += 1

    return result


def sort_result(result, order):
    """support -- по убыванию поддержки, lex -- лексикографический."""
    if order == "support":
        return sorted(result, key=lambda x: (-x[1], x[0]))
    if order == "lex":
        return sorted(result, key=lambda x: x[0])
    raise ValueError(f"неизвестный порядок: {order}")


def parse_support(value):
    """'0.03', '3%' и '3' -> 0.03"""
    value = value.strip().replace(",", ".")
    if value.endswith("%"):
        value = float(value[:-1]) / 100
    else:
        value = float(value)
        if value > 1:
            value /= 100
    if not 0 < value <= 1:
        raise ValueError("support threshold has to be in (0, 1]")
    return value


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return

    path = sys.argv[1]
    min_support = parse_support(sys.argv[2])
    order = sys.argv[3] if len(sys.argv) > 3 else "support"
    out_path = sys.argv[4] if len(sys.argv) > 4 else None

    transactions = load_transactions(path)
    result = sort_result(apriori(transactions, min_support), order)

    print(f"Transactions: {len(transactions)}, support threshold: {min_support:.2%}, "
          f"frequent sets found: {len(result)}")
    for itemset, support in result:
        print(f"{support:8.4f}  {{{', '.join(itemset)}}}")

    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["size", "support", "itemset"])
            for itemset, support in result:
                w.writerow([len(itemset), f"{support:.6f}", "; ".join(itemset)])
        print(f"Результат сохранён в {out_path}")


if __name__ == "__main__":
    main()