import numpy as np

# -*- coding: utf-8 -*-
"""
polyFeatureVector and plotDecisionBoundary1
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd


def polyFeatureVector(x1, x2, degree):
    """

    """
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    Xp = np.ones(shape=(x1[:, 0].size, 1))

    m, n = Xp.shape

    for j in range(1, degree + 1):
        for k in range(j + 1):
            p = (x1 ** (j - k)) * (x2 ** k)
            Xp = np.append(Xp, p, axis=1)
    return Xp


def plotDecisionBoundary1(theta, X, y, d, Lambda):
    x1 = X[:, 1]
    x2 = X[:, 2]
    plt.plot()
    fig1 = plt.plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'ro', x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go')
    plt.xlabel('x1'), plt.ylabel('x2')
    plt.title(f'data; Lambda={np.around(Lambda, decimals=3)}')
    plt.grid()
    plt.plot()
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(polyFeatureVector(np.array([u[i]]), np.array([v[j]]), d), theta)
    z = np.transpose(z)
    fig1 = plt.contour(u, v, z, levels=[0], linewidth=2).collections[0]


def map_feature(x1, x2, degree=6):
    '''
    Maps a two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    x1, x2, x1 ** 2, x2 ** 2, x1*x2, x1*x2 ** 2, etc...
    The inputs x1, x2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    X = np.ones(shape=(x1[:, 0].size, 1))

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            print(f'x1**{i - j}, x2**{j}')
            X = np.append(X, r, axis=1)

    return X


def grad_decent(X, y, Theta, alpha=1, max_iter=10000, epsilon=0.00001, Lambda=0):
    J_theta = np.zeros(max_iter)
    for i in range(max_iter):
        J, dJ = compute_cost(X, y, Theta, Lambda)
        Theta -= alpha * dJ
        J_theta[i] = J
        if i > 0 and np.abs(J_theta[i] - J_theta[i - 1]) < epsilon:
            print(f"INFO: EARLY STOP AFTER {i} iterations")
            return J_theta[:i], Theta
    return J_theta, Theta


def compute_cost(X, y, Theta, Lambda=0):
    m = X.shape[0]
    n = X.shape[1]

    h_Theta = sigmoid(np.dot(X, Theta))

    penalty = 0
    d_penalty = 0
    if Lambda != 0:
        penalty = (Lambda / 2 * m) * (np.dot(Theta.T, Theta))
        d_penalty = (Lambda / m) * Theta

    J = -1 / m * (np.dot(y.T, np.log(h_Theta)) + np.dot((1 - y).T, np.log(1 - h_Theta))) + penalty

    dJ = (1 / m) * np.dot(X.T, h_Theta - y) + d_penalty

    return J, dJ


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
