# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:33:53 2022
Perceptron exercise
@author: YL
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_points_and_boundary(X, y, theta, plot_title, b=0, freeze=2.0, final=False):
    """"

    """
    ind = 1
    x1_min = 1.1 * X[:, ind].min()
    x1_max = 1.1 * X[:, ind].max()
    x2_min = - (b + theta[0] * x1_min) / theta[1]
    x2_max = - (b + theta[0] * x1_max) / theta[1]
    x1lh = np.array([x1_min, x1_max])
    x2lh = np.array([x2_min, x2_max])
    x1 = X[:, 0]
    x2 = X[:, 1]
    plt.plot(x1[y[:, 0] == 1], x2[y[:, 0] == 1], 'go',
             x1[y[:, 0] == -1], x2[y[:, 0] == -1], 'rx',
             x1lh, x2lh, 'b-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('data')
    plt.grid(axis='both')
    plt.draw()
    plt.pause(0.05)
    if not final:
        plt.clf()


def perceptron_train(X, y, plotflag, max_iter):
    """
    perceptron_train implements perceptron learning algorithm
    (linear classifiers where the decision boundary is through the origin)
    Input arguments
    ----------
    X : data matrix, where each row is one observation
    y : labels (1 or -1)
    plotflag : 1 if to plot
    max_iter : maximum number of iterations

    Returns
    -------
    theta, k - number of iterations (until a decision boundary classify all the
                                     samples correctly)

    """

    num_correct = 0
    mat_shape = X.shape
    if len(mat_shape) > 1:
        nrow = mat_shape[0]
        ncol = mat_shape[1]
    else:
        X = X.reshape(X.shape[0], 1)

    current_index = 0
    theta = np.zeros((ncol, 1))
    b = 0
    j = 0
    k = 0
    is_first_iter = 1
    while num_correct < nrow and k < max_iter:
        j = j + 1
        xt = X[current_index, :]
        xt = xt.reshape(xt.shape[0], 1)
        yt = y[current_index]
        # ---------------------------------------------------------------------
        a = yt * (np.dot(theta.T, xt) + b)  # your code here (one line). if the sign of the hypothesis function
        # is not equal yt it should be negative, otherwise it should be
        # positive. Include here the bias term b in your code and assign
        # b = 0.
        # ---------------------------------------------------------------------
        if is_first_iter == 1 or a < 0:
            # -----------------------------------------------------------------
            ####### your code here (3 lines)
            theta = theta + yt * xt
            b = b + yt
            num_correct = 0  # it should be zeroed after each error
            k += 1  # this counts the iterations (i.e. the number of error occurences)
            #########
            # -----------------------------------------------------------------
            is_first_iter = 0
            if plotflag == 1:
                plot_points_and_boundary(X, y, theta, b)
        else:
            num_correct += 1
            print(num_correct)
        current_index += 1
        if current_index > nrow - 1:
            current_index = 1
    return theta, k, b


# 0 producing separable dataset
def make_decentralized_dataset():
    x0 = np.random.randn(50, 2) + 1.8 * np.ones((50, 2))
    x1 = np.random.randn(50, 2) + 5.3 * np.ones((50, 2))
    X = np.concatenate((x0, x1), axis=0)
    y = np.ones((100, 1))
    y[50:] = -1
    plt.plot(y)
    plt.plot(x0[:, 0], x0[:, 1], 'ro', x1[:, 0], x1[:, 1], 'bx')
    return X, y


# -----------------------------------------------------------------------------
##### your code here - load the data according to the instructions in the exercise
def read_centralized_dataset():
    npzfile = np.load("Perceptron_exercise_2.npz")
    sorted(npzfile.files)
    X = npzfile['arr_0']
    y = npzfile['arr_1']
    return X, y


# -----------------------------------------------------------------------------
def main():
    plotflag = 1
    max_iter = 300
    X, y = read_centralized_dataset()
    thata, k, _ = perceptron_train(X, y, plotflag, max_iter)
    X, y = make_decentralized_dataset()
    thata, k, b = perceptron_train(X, y, plotflag, max_iter)
    plot_points_and_boundary(X, y, theta=thata, plot_title='Decentralize data set', b=b, final=True)
    plt.pause(5)
    print('k =', k)


if __name__ == '__main__':
    main()
