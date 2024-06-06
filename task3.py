import numpy as np
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
from math import log, e
import time
from scipy.stats.sampling import SimpleRatioUniforms


def my_distribution(x):
    return (e ** (-abs(x))) * 1 / 2 / (1 - e ** (-1))


n = [3, 100, 1000, 10000]


def plot(test_num, am, stime, data):
    plt.figure(figsize=(15, 9))

    plt.title('PDF and Histogram of Random Samples')
    plt.xlabel('x')
    plt.ylabel('Density')

    count, bins, ignored = plt.hist(data, range=(-1, 1), bins=30, alpha=0.6, color='b',
                                    density=True, label='Histogram of samples')

    plt.plot(bins, list(map(lambda x: my_distribution(x), bins)), linewidth=2,
             color='r')

    info_text = f'Test №{test_num}\nNumber of samples: {am}\nTime: {stime[1] - stime[0]:4f}\nAverage time: {(stime[1] - stime[0]) / am:.4f}'
    plt.text(0.1, 1.03, info_text, transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.show()


def test(num, func):
    global n
    for i in n:
        start_time = time.time()
        random_values = func(i)
        end_time = time.time()
        plot(num, i, [start_time, end_time], random_values)


class MyContinuous(rv_continuous):

    def _pdf(self, x, *args):
        return my_distribution(x)


def func1(sz):
    custom_dist = MyContinuous(a=-1.0, b=1.0, name='custom_dist')
    return custom_dist.rvs(size=sz)


def F_1(x):
    if x <= 1/2:
        return log(x * 2 * (1 - e ** -1) + 1 / e)
    else:
        return -log(1 - (x - 1 / 2) * 2 * (1 - e ** -1))


def func2(sz):
    random = np.random.uniform(0, 1, sz).tolist()
    return list(map(F_1, random))


def func3(sz):
    urng = np.random.default_rng()
    dist = MyContinuous()
    rng = SimpleRatioUniforms(dist, mode=0,
                              pdf_area=np.sqrt(2 * np.pi),
                              random_state=urng)
    return rng.rvs(sz)


if __name__ == "__main__":
    print(F_1(0))
    test(1, func1)
    test(2, func2)
    test(3, func3)
