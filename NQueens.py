from sys import stdout
from time import time
from joblib import Parallel, delayed
import numpy as np


def print_t(t, out=stdout):
    N = len(t)
    for i in range(N):
        print(str([int(t[i] == j) for j in range(N)]).
              replace("[", "").replace("]", "").replace(",", ""), file=out)


def h(t):
    N = len(t)
    h = 0
    for i in range(N - 1):

        c = [0, 0, 0, 0]
        for j in range(N - i):
            # top principal diagonals
            if t[j] == j + i:
                c[0] += 1

            # top secondary diagonals
            if t[j] == N - j - i - 1:
                c[1] += 1

            if i > 0:
                # bottom principal diagonals
                if t[j + i] == j:
                    c[2] += 1

                # bottom secondary diagonals
                if t[j + i] == N - j - 1:
                    c[3] += 1
        h += np.sum([i - 1 if i > 1 else 0 for i in c])
    return h


def individual_h(t, k):
    N = len(t)
    l = t[k]
    c1 = c2 = 0

    # print("pos", k, l)
    if k < l:  # diagonal principal superior
        i = l - k
        # print("i", i)
        for j in range(N - i):
            # print(j, j + i)
            if t[j] == j + i:
                c1 += 1

        if k + l >= N:  # diagonal secundária inferior
            # print("NOT1")
            i = k + l
            for j in range(N + N - i - 1):
                # print(j + i - N + 1, N - j - 1)
                if t[j + i - N + 1] == N - j - 1:
                    c2 += 1
        else:  # diagonal secundária superior
            # print("OK1")
            i = k + l
            for j in range(i + 1):
                # print(i - j, j)
                if t[i - j] == j:
                    c2 += 1
    else:  # diagonal principal inferior
        i = k - l
        # print("i", i)
        for j in range(N - i):
            # print(j + i, j)
            if t[j + i] == j:
                c1 += 1

        if k + l >= N:  # diagonal secundária inferior
            # print("NOT2")
            i = k + l
            for j in range(N + N - i - 1):
                # print(j - N + i + 1, N - j - 1)
                if t[j - N + i + 1] == N - j - 1:
                    c2 += 1
        else:  # diagonal secundária superior
            # print("OK2")
            i = k + l
            for j in range(i + 1):
                # print(i - j, j)
                if t[i - j] == j:
                    c2 += 1

    return int(c1 > 1) + int(c2 > 1)


def individual_h2(t, k, l):
    N = len(t)
    h = 0

    for i in range(N):
        if abs(i - k) == abs(t[i] - t[k]):
            h += 1
        if abs(i - l) == abs(t[i] - t[l]):
            h += 1

    return h - 2


def partial_v(t, i):
    for j in range(i):
        if abs(i - j) == abs(t[i] - t[j]):
            return False


def v(t):
    N = len(t)
    return all(Parallel(n_jobs=-1)(delayed(partial_v)(t, i) for i in range(1, N)))


def partial_h2(t, i):
    c = 0
    for j in range(i):
        if abs(i - j) == abs(t[i] - t[j]):
            c += 1
    return c


def h2(t):
    N = len(t)
    # h = Parallel(n_jobs=-1)(delayed(partial_h2)(t, i) for i in range(1, N))
    h = 0
    conflicts = []
    for i in range(1, N):
        for j in range(i):
            if abs(i - j) == abs(t[i] - t[j]):
                h+=1
                conflicts.append(i)
    return h, conflicts


def partial_h3(t, i):
    for j in range(i):
        if abs(i - j) == abs(t[i] - t[j]):
            return 1
    return 0


def h3(t):
    N = len(t)
    h = Parallel(n_jobs=-1)(delayed(partial_h3)(t, i) for i in range(1, N))
    conflicts = set()
    for i in range(N-1):
        if h[i]:
            conflicts.add(i+1)
    return np.sum(h), list(conflicts)


if __name__ == "__main__":
    N = 4000
    t = np.random.permutation(N)

    # print(t)
    # print_t(t)

    start = time()
    h_t = h(t)
    print("Heuristic:", h_t)
    elapsed = time() - start
    print("Elapsed (sec):", elapsed)

    start = time()
    h_t = h2(t)
    print("Heuristic 2:", h_t)
    elapsed = time() - start
    print("Elapsed (sec):", elapsed)

    start = time()
    h_t = h3(t)
    print("Heuristic 3:", h_t)
    elapsed = time() - start
    print("Elapsed (sec):", elapsed)

    start = time()
    v_t = v(t)
    print("Valid:", v_t)
    elapsed = time() - start
    print("Elapsed (sec):", elapsed)
