"""
Helper functions
"""
from bisect import bisect_left


def sort_a_by_b(a, b):
    res = []
    a_sorted = sorted(a)
    for item in b:
        found = bisect_left(a_sorted, item)
        if found < len(a_sorted) and a_sorted[found] == item:
            del a_sorted[found]
            res.append(item)

    for item in a_sorted:
        res.append(item)

    return res
