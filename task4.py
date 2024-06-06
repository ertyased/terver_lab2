import numpy as np
from scipy.special import erf

eps = 0.01
delta = 0.05
am = 10
lam = [1 + 4 / am * i for i in range(am + 1)]



def test(name, func, lambd, test_am):
    n = func(lambd)
    good = 0
    for i in range(test_am):
        res = np.random.exponential(scale=1 / lambd, size=n).tolist()  # берем 1 / lambda потому что в numpy
        sum_res = sum(res)  # экспоненциальная функция от 1 / lambda
        if abs(sum_res / n - 1 / lambd) < eps:
            good += 1

    print(f'test {name}  for epsilon: {eps}, delta: {delta}, and lambda: {lambd}. Found n is {n}. amount of good tests: {good}\n' +
          f'delta result is: {1 - good / test_am}')


def n_cheb(lambd):
    return int(1 / (lambd * lambd * eps * eps * delta))

def test_cheb(lambd, test_am):
    test("Chebyshev", n_cheb, lambd, test_am)


def find_n(lambd):
    l = 1
    r = 10**9
    while r - l > 1:
        mid = (r + l) // 2
        if erf(mid**0.5 * lambd * eps) > 1 -  delta:
            r = mid
        else:
            l = mid
    return r


def test_clt(lamb, test_am):
    test("Central Limit Theorem", find_n, lamb, test_am)

if __name__ == "__main__":
    for i in lam:
        test_cheb(i, 100)

    for i in lam:
        test_clt(i, 100)