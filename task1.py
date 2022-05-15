import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def plot_result(X, y, theta, J_theta, alpha=None, type='linear', title=None):
    X1 = X[:, 1]
    X2 = X[:, 2]

    plt.subplot(1, 2, 1)

    # Plot dataset labels
    plt.plot(X1[y[:, 0] == 0], X2[y[:, 0] == 0], 'go')
    plt.plot(X1[y[:, 0] == 1], X2[y[:, 0] == 1], 'ro')
    plt.xlabel('x1'), plt.ylabel('x2')
    # ax.scatter(X1, X2)
    if type == 'linear':
        y_line = [-(theta[0][0] + theta[1][0] * np.min(X1)) / theta[2][0],
                  -(theta[0][0] + theta[1][0] * np.max(X1)) / theta[2][0]]
        x_line = [np.min(X1), np.max(X2)]
    if type == 'quadratic':
        x_line = np.linspace(np.min(X1), np.max(X1), X1.shape[0])
        y_line = -(theta[0][0] + theta[1][0] * x_line + theta[3][0] * x_line ** 2) / theta[2][0]
    plt.ylim(np.min(X2) - 1, np.max(X2) + 1)
    plt.plot(x_line, y_line)

    plt.subplot(1, 2, 2)
    if alpha is not None:
        plt.title(f'{alpha}')
    if title is not None:
        plt.title(title)
    plt.plot(J_theta)
    plt.draw()
    plt.pause(0.001)
    plt.clf()


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def compute_cost(X, y, Theta, Lambda=0):
    m = X.shape[0]
    n = X.shape[1]

    h_Theta = sigmoid(np.dot(X, Theta))

    penalty = 0
    penalty_tag = 0
    if Lambda != 0:
        penalty = Lambda / (2 * m) * np.sum(Theta ** 2)
        penalty_tag = (Lambda / m) * Theta

    J = -1 / m * (np.dot(y.T, np.log(h_Theta)) + np.dot((1 - y).T, np.log(1 - h_Theta))) + penalty  # Explore this

    dJ = (1 / m) * np.dot(X.T, h_Theta - y) + penalty_tag

    return J, dJ


def grad_decent(X, y, Theta, alpha=1, max_iter=10000, epsilon=0.00001):
    J_theta = np.zeros(max_iter)
    for i in range(max_iter):
        J, dJ = compute_cost(X, y, Theta)
        Theta -= alpha * dJ
        J_theta[i] = J
        if i > 0 and np.abs(J_theta[i] - J_theta[i - 1]) < epsilon:
            print(f"INFO: EARLY STOP AFTER {i} iterations")
            return J_theta[:i], Theta
    return J_theta, Theta


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def main():
    """PREPARE INIT DATA"""
    ####################################################################################################################
    data = pd.read_csv('email_data_1.csv')
    data = data.to_numpy()

    X = data[:, :2]
    y = data[:, 2]
    y = y.reshape((y.shape[0], 1))

    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    Theta = np.zeros((X.shape[1], 1))
    ####################################################################################################################

    """LINEAR"""
    ####################################################################################################################
    J_theta, Theta = grad_decent(X, y, Theta)
    plot_result(X, y, Theta, J_theta, title='linear')
    ####################################################################################################################

    """QUADRATIC"""
    ####################################################################################################################
    X_quadratic = np.concatenate((X, X[:, 1].reshape(X.shape[0], 1) ** 2), axis=1)
    Theta_quadratic = np.zeros((X_quadratic.shape[1], 1))
    J_theta_quadratic, Theta_quadratic = grad_decent(X_quadratic, y, Theta_quadratic)
    plot_result(X_quadratic, y, Theta_quadratic, J_theta_quadratic, type='quadratic', title='quadratic')
    ####################################################################################################################

    """TRYING OUT ALPHAS"""
    ####################################################################################################################
    alphas = np.linspace(4, 10, 100)
    n = np.empty_like(alphas)
    J_thetas = list()

    for i, alpha_i in enumerate(alphas):
        J_theta_quadratic, Theta_quadratic = grad_decent(X_quadratic, y, Theta_quadratic, alpha=alpha_i)
        J_thetas.append(J_theta)
        n[i] = len(J_theta_quadratic)
    ####################################################################################################################

    """PLOTTING NUM ITERATIONS TILL CONVERGENCE FOR DIFFERENT ALPHAS ALONG WITH THE RESULTING J_Theta"""
    ####################################################################################################################
    optimal_alpha = np.around(alphas[np.argmin(n)], decimals=3)
    min_num_iter = n.min()
    plt.subplot(1, 2, 1)
    plt.scatter(optimal_alpha, min_num_iter, c='r')
    plt.title(f"for alpha = {optimal_alpha} min num iterations = {min_num_iter}")
    plt.xlabel('alpha')
    plt.ylabel('num iterations')
    plt.plot(alphas, n)
    plt.subplot(1, 2, 2)
    plt.title(f'J_theta for alpha = {optimal_alpha}')
    plt.plot(J_thetas[int(np.where(alphas == alphas[np.argmin(n)])[0])])
    plt.show()
    ####################################################################################################################

    """TESTING THE MODEL"""
    ####################################################################################################################

    test_data = pd.read_csv('email_data_test_2021.csv')

    # USING TRAINED LINEAR AND QUADRATIC FITTING MODELS FOR UNSEEN DATA
    test_data['linear_predict'] = Theta[0][0] + test_data['x1'] * Theta[1][0] + test_data['x2'] * Theta[2][0]
    test_data['quadratic_predict'] = Theta_quadratic[0][0] + \
                                     test_data['x1'] * Theta_quadratic[1][0] + \
                                     test_data['x2'] * Theta_quadratic[2][0] + \
                                     (test_data['x1'] ** 2) * Theta_quadratic[3][0]

    # CONVERTING TO BINARY DECISION
    test_data['linear_predict'] = test_data['linear_predict'].apply(lambda x: 1 if x > 0 else 0)
    test_data['quadratic_predict'] = test_data['quadratic_predict'].apply(lambda x: 1 if x > 0 else 0)

    # COUNTING MODELS HITS
    linear_predictor_hits = test_data.apply(lambda x: True if x['linear_predict'] == x['ytest'] else False, axis=1)
    linear_predictor_hits = len(test_data[linear_predictor_hits == True].index)

    quardratic_predictor_hits = test_data.apply(lambda x: True if x['quadratic_predict'] == x['ytest'] else False,
                                                axis=1)
    quardratic_predictor_hits = len(test_data[quardratic_predictor_hits == True].index)

    # GETTING THE ACCURACY PERCENTAGE
    n_samples = test_data.shape[0]
    print(f'Linear model accuracy = {linear_predictor_hits / n_samples * 100}')
    print(f'Quadratic model accuracy = {quardratic_predictor_hits / n_samples * 100}')
    ####################################################################################################################


if __name__ == '__main__':
    main()
