# -*- coding: utf-8 -*-
"""
polyFeatureVector and plotDecisionBoundary1
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

# Supress warnings for this module
warnings.filterwarnings("ignore")


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


def plotDecisionBoundary1(theta, X, y, d):
    x1 = X[:, 1]
    x2 = X[:, 2]
    fig1 = plt.plot(x1[y[:, 0] == 0], x2[y[:, 0] == 0], 'ro', x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go')
    plt.xlabel('x1'), plt.ylabel('x2')
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
