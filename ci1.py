import pprint
import numpy as np
from typing import Any
def fuzzy_union(a: dict[Any, float], b: dict[Any, float]) -> dict[Any, float]:
    return {
        x: max(a.get(x, 0), b.get(x, 0))
        for x in set(a).union(set(b))
    }

def fuzzy_intersection(a: dict[Any, float], b: dict[Any, float]) -> dict[Any, float]:
    return {
        x: min(a.get(x, 0), b.get(x, 0))
        for x in set(a).union(set(b))
    }

def fuzzy_complement(a: dict[Any, float]) -> dict[Any, float]:
    return {
        x: 1 - deg
        for x, deg in a.items()
    }

def fuzzy_difference(a: dict[Any, float], b: dict[Any, float]) -> dict[Any, float]:
    b_c = fuzzy_complement(b)
    return fuzzy_intersection(a, b_c)

def fuzzy_cartesian_product(a: dict[Any, float], b: dict[Any, float]) -> list[list[float]]:
    relation = [[0.0 for _ in range(len(b))] for _ in range(len(a))]

    for ix, x in enumerate(a.keys()):
        for iy, y in enumerate(b.keys()):
            relation[ix][iy] = min(a[x], b[y]) 

    return relation

def min_max_composition(
        r: dict[Any, float] | list[list[float]], 
        s: dict[Any, float] | list[list[float]], 
) -> dict[Any, float] | list[list[float]]:
    relation = []

    print()
    print(r)
    print(list(zip(*s)))

    for x in r:
        row = []
        for y in zip(*s):
            z = list(map(min, zip(x, y)))
            row.append(max(z))

        relation.append(row)

    return relation


if __name__ == "__main__":
    # Example Fuzzy Sets
    A = {'x1': 0.2, 'x2': 0.7, 'x3': 1.0}
    B = {'y1': 0.5, 'y2': 0.4, 'y3': 0.8}

    R = fuzzy_cartesian_product(A, B)
    print("\nFuzzy Relation R (A x B):")
    pprint.pprint(R)

    C = {'z1': 0.6, 'z2': 0.9}

    S = fuzzy_cartesian_product(B, C)
    print("\nFuzzy Relation S (B x A):")
    pprint.pprint(S)
    
    T = min_max_composition(R, S)
    print("\nMin-max composition")
    pprint.pprint(T)
