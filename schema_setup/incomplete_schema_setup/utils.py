import numpy as np
from sklearn.linear_model import LinearRegression


def compute_correlated_vector(y, rho, probability_distribution=False, normalize_input=False):
    """
    Given a vector y returns a second vector which has a pearson correlation of rho

    Credits: https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables

    :param y: original vector y
    :param rho: target pearson correlation
    :param probability_distribution: whether the resulting vector should be a valid probability distribution
    :param normalize_y:
    :return:
    """
    assert not np.any(np.isnan(y))

    # normalize to be in range [0,1]
    if normalize_input:
        y -= np.min(y)
        y /= np.max(y)

    x = np.random.uniform(size=len(y))
    # Find the residuals
    y_features = y.reshape(-1, 1)
    y_perp = x - LinearRegression().fit(y_features, x).predict(y_features)

    # compute correlated vector
    corr = rho * np.std(y_perp) * y + y_perp * np.std(y) * np.sqrt(1 - rho ** 2)

    if probability_distribution:
        if np.any(corr < 0):
            corr -= np.min(corr)
        corr /= np.sum(corr)
        assert np.isclose(np.sum(corr), 1)
        assert np.all(corr >= 0)

    return corr


def discretize(x, steps):
    # if np.any(x < 0):
    x -= np.min(x)
    x *= 1 / np.max(x)
    x *= (steps - 1)
    x = np.ceil(x).astype(int).astype(float)
    x /= (steps - 1)

    assert len(np.unique(x)) <= steps

    return x


def zipf(a, min, max, size=None):
    v = np.arange(min, max + 1)
    p = 1.0 / np.power(v, a)
    p /= np.sum(p)

    return np.random.choice(v, size=size, replace=True, p=p).astype(np.float)


def common_element(x, least_common=False):
    unique, counts = np.unique(x, return_counts=True)
    if least_common:
        return unique[np.argmin(counts)]
    return unique[np.argmax(counts)]
