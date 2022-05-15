import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from map_feature import map_feature
import thlearn


def linear(X_orig, y, m):
    # linear model
    X = np.concatenate((np.ones((m, 1)), X_orig), axis=1)
    Theta = np.zeros(X[0].shape)

    J_theta, Theta = thlearn.grad_decent(X, y, Theta)
    # plot_result(X, y, Theta, J_theta)
    plt.plot(J_theta)
    plt.title("J_Theta (email classification)")
    plt.show()

    #####FINDING TWO POINTS ON THE PLANE TO DRAW THE LINE#########
    x2_min = -(Theta[0] + Theta[1] * X[:, 1].min()) / Theta[2]
    x2_max = -(Theta[0] + Theta[1] * X[:, 1].max()) / Theta[2]

    plt.scatter(X[:, 1][y == 0], X[:, 2][y == 0], c='r')
    plt.scatter(X[:, 1][y == 1], X[:, 2][y == 1], c='g')
    plt.plot([X[:, 1].min(), X[:, 1].max()], [x2_min, x2_max])
    plt.title("email data with linear model")
    plt.show()


def polynomial(X_polynomial, y, Lambda=0, animate=False):
    degree = 6

    Theta = np.zeros((X_polynomial.shape[1], 1))

    J_theta, Theta = thlearn.grad_decent(X_polynomial, y, Theta, alpha=0.1, Lambda=Lambda)

    thlearn.plotDecisionBoundary1(Theta, X_polynomial, y, degree, Lambda=Lambda)
    if animate:
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
    return J_theta, Theta


def explore_regulation_coefficient(X_orig, y):
    Lambdas = np.linspace(0, 30, 60)
    best_score = 0
    best_Lambda = 0
    best_Theta: np.ndarray
    for Lambda in Lambdas:
        score_i, Theta = score_model(X_orig, y, Lambda=Lambda)
        if score_i > best_score:
            best_score = score_i
            best_Lambda = Lambda
            best_Theta = Theta
    return best_Lambda, best_score, best_Theta


def score_model(X_orig, y, Lambda):
    x1 = X_orig[:, 0]
    x2 = X_orig[:, 1]
    X_polynomial = map_feature(x1, x2, degree=6)  # transform the data to the polynomial of degree 6

    J_theta, Theta = polynomial(X_polynomial, y, Lambda=Lambda)  # regression with transformed data

    H_theta = np.dot(X_polynomial, Theta)  # compute the hypothesis

    #count samples, correctly classified by the model
    hypothesis_hits = np.sum([y[np.where(H_theta > 0)] == 1]) + np.sum([y[np.where(H_theta < 0)] == 0])

    return hypothesis_hits / y.shape[0], Theta


def main():
    """IF YOU'RE CHECKING THIS ASSIGNMENT, PLEASE NOTE THAT IN ORDER TO CANCEL THE ANIMATION,
    MANUALLY DROP THE FLAG `animate` IN `def polynomial()`"""
    Xdata = pd.read_csv("email_data_2.csv")

    data = Xdata.to_numpy()
    X_orig = data[:, 0:2]
    y = data[:, 2]

    m = y.size
    # linear(X_orig, y, m)
    # conclusion:
    print("INFO: THE DATA SET IS NOT LINEARLY SEPARABLE")

    """`explore_regulation_coefficient(X, y)` calls to regulative regression tools with data, transformed with 
    respect to higher degree polynomial. It invokes regression multiple times with different regulation coefficients 
    (Lambda) and compares regression's prediction to actual result. `optimal_lambda` is returned for the most accurate 
    prediction """
    y = np.atleast_2d(y).T
    optimal_lambda, optimal_score, optimal_Theta = explore_regulation_coefficient(X_orig, y)
    print(f'Optimal Lambda = {optimal_lambda} with accuracy of {optimal_score}')

    test_set = pd.read_csv("email_data_3_2021.csv")
    test_data = test_set.to_numpy()
    x1 = test_data[:, 0]
    x2 = test_data[:, 1]
    test_data_polynomial = map_feature(x1, x2, degree=6)
    y = test_data[:, 2]
    test_data = test_data[:, 0:2]
    y = np.atleast_2d(y).T
    H_theta = np.dot(test_data_polynomial, optimal_Theta)
    accuracy = (np.sum([y[np.where(H_theta > 0)] == 1]) + np.sum([y[np.where(H_theta < 0)] == 0]))/m
    print(f'Accuracy for email_data_3_2021 = {accuracy}')


if __name__ == '__main__':
    main()
