import numpy as np


def _sigmoid(x, threshold=500):
    return 1 / (1 + np.exp(np.clip(-x, -threshold, threshold)))


def normal_scaled(size,
                  num_features=21,
                  loc=0,
                  max_exponent=10.0,
                  seed=None,
                  return_probs=False):
    scales = 2.0 ** ((np.arange(num_features) / (num_features - 1) - 0.5) * 2 * max_exponent)
    if seed is not None:
        np.random.seed(seed)
    x = np.random.normal(loc=loc,
                         scale=scales,
                         size=(size, num_features))
    w = np.ones(x.shape[1], dtype=np.float32) / (scales)
    # np.random.normal(size=x.shape[1]) / (scales+_SMALL_NUMBER)
    probs = _sigmoid(np.matmul(x, w))
    labels = np.int64(np.random.binomial(1, probs))
    if return_probs:
        return x, labels, probs

    return x, labels


def normal_dist_outliers(
        size,
        num_features,
        loc=0,
        seed=None,
        outliers_ratio=0.001,
        outlier_scale=100,
        return_probs=False):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.normal(loc=loc,
                         scale=1.0,
                         size=(size, num_features))

    num_outliers = int(len(x) * outliers_ratio)
    outlier_indices = np.repeat(np.arange(num_features), num_outliers)

    x[np.random.randint(0, len(x), num_features * num_outliers), outlier_indices] *= np.random.normal(loc=outlier_scale,
                                                                                                      size=num_features * num_outliers)

    w = np.random.normal(size=x.shape[1]) / num_features ** 0.5

    probs = _sigmoid(np.matmul(x, w))
    labels = np.int64(np.random.binomial(1, probs))

    if return_probs:
        return x, labels, probs

    return x, labels
