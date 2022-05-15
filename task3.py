import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn
from sklearn import datasets
import numpy as np
import thlearn

TEST = False
TRAIN = True


def choose_and_plot_subset(X, y, plot=True, stage=TRAIN):
    # subset of 30 first samples of each class
    rows_to_remove_train = list(range(30, 50)) + list(range(80, 100)) + list(range(130, X.shape[0]))
    if stage == TRAIN:
        rows_to_remove = rows_to_remove_train
    elif stage == TEST:
        rows_to_remove = list(
            set(range(X.shape[0])) - set(rows_to_remove_train))  # taking all rows except those for training

    X_sub = np.delete(X, rows_to_remove, axis=0)
    y_sub = np.delete(y, rows_to_remove, axis=0)

    ######ASSERTION - 30 samples from each class ########
    if stage == TRAIN:
        for i in range(3):
            assert np.sum(np.array(y_sub) == i) == 30

    if plot:
        plot_data(X_sub, y_sub, "first 30 samples of each class")
    return X_sub, y_sub


def plot_data(X, y, title, Thetas=None):
    plt.figure(2, figsize=(8, 6))
    plt.clf()

    colors = ['r', 'g', 'b']
    labels = ['setosa', 'versicolor', 'virginica']
    cmap = matplotlib.colors.ListedColormap(colors)

    if X.shape[1] == 2:  # for plotting the raw data
        x1slice = np.s_[:, 0]
        x2slice = np.s_[:, 1]
    elif X.shape[1] == 3:  # for plotting the data with expanded dimensions (column of ones stacked vertically)
        x1slice = np.s_[:, 1]
        x2slice = np.s_[:, 2]
    else:
        raise ValueError(f'Shape of X is {X.shape}, accepted dimensions are (nrows,2) or (nrows, 3)')

    plt.scatter(X[x1slice], X[x2slice], c=y, cmap=cmap)

    plt.title(title)
    patches = [matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
    plt.legend(handles=patches)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    if Thetas is not None:
        x, lines = get_lines(X, Thetas, x1slice, x2slice)
        for line in lines:
            plt.plot(x, line)
    plt.show()


def get_lines(X, Thetas, x1slice, x2slice):
    x_min, x_max = X[x1slice].min() - 5, X[x2slice].max() + 5
    n_classes = len(Thetas)

    # line_i separates class i form the union of the rest of the classes
    lines = [
        [-(Thetas[i][0] + Thetas[i][1] * x_min) / Thetas[i][2], -(Thetas[i][0] + Thetas[i][1] * x_max) / Thetas[i][2]]
        for i in range(n_classes)
    ]
    return [x_min, x_max], lines


def one_vs_all_classification(X_sub, y_sub):
    m = X_sub.shape[0]
    n_classes = 3
    # X_sub = (X_sub - np.mean(X_sub, axis=0)) / np.std(X_sub, axis=0)  # normalize the data
    X_sub = np.c_[np.ones(m), X_sub]
    # X_sub = np.concatenate((np.ones((X_sub.shape[0], 1)), X_sub), axis=1)

    Thetas = [np.zeros((X_sub.shape[1], 1)) for _ in
              range(n_classes)]  # 3 coefficient vectors for each classification step
    J_theta = list()
    for class_i in range(n_classes):
        y_class_i = np.copy(y_sub)
        if class_i == 0:
            y_class_i[y_sub != class_i] = 1  # class_0 labeled as 0, Union({class_1},{class_2}) labeled as 1
        if class_i == 1:
            y_class_i[y_sub != class_i] = 0  # class_1 labeled as 1, Union({class_0},{class_2}) labeled as 0
        if class_i == 2:
            y_class_i[y_sub == class_i] = 1  # class_2 labeled as 1,
            y_class_i[y_sub != class_i] = 0  # Union({class_0},{class_1}) labeled as 0

        J_theta_i, Thetas[class_i] = thlearn.grad_decent(X_sub, y_class_i, Thetas[class_i], Lambda=1)
        J_theta.append(J_theta_i)
    plot_data(X_sub, y_sub, Thetas=Thetas, title='one vs all logistic regression classification')
    return Thetas


def score_model(H_theta, y):
    hypothesis_hits_0 = (np.sum([y[np.where(H_theta[:, 0] <= 0)] == 0])
                         + np.sum([y[np.where(H_theta[:, 0] >= 1)] >= 1])) / y.shape[0]

    hypothesis_hits_1 = (np.sum([y[np.where(H_theta[:, 1] <= 0)] == 0])
                         + np.sum([y[np.where(H_theta[:, 1] <= 0)] == 2])
                         + np.sum([y[np.where(H_theta[:, 1] >= 0)] == 1])) / y.shape[0]

    hypothesis_hits_3 = (np.sum([y[np.where(H_theta[:, 2] >= 0)] == 2])
                         + np.sum([y[np.where(H_theta[:, 2] <= 0)] <= 1])) / y.shape[0]

    accuracy = (hypothesis_hits_0 + hypothesis_hits_1 + hypothesis_hits_3) / 3

    return accuracy


def main():
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]  # we only take the first two features.
    y = np.atleast_2d(iris.target).T

    ######################PLOTTING TRAINING DATA SET  ###################
    X_sub, y_sub = choose_and_plot_subset(X, y, plot=False, stage=TRAIN)
    X_sub = thlearn.normalize(X_sub)
    #####################################################################

    #####################LEARNING ON TRAINING DATA SET###################
    Thetas = one_vs_all_classification(X_sub, y_sub)
    Thetas = np.array(Thetas).reshape(3, -1)  # preparing `list` `Thetas` for dot product
    #####################################################################

    #####################EXTRACTING TESTING DATA SET#####################
    X_sub_test, y_sub_test = choose_and_plot_subset(X, y, plot=False, stage=TEST)
    X_sub_test = np.c_[np.ones(X_sub_test.shape[0]), X_sub_test]
    #####################################################################

    ###COMPUTING THE HYPOTHESIS OF TESTING DATA SET, USING PREVIOUSLY LEARNT PARAMETERS THETA#####################
    H_theta = np.dot(X_sub_test, Thetas)
    ##############################################################################################################

    #############SCORING THE HYPOTHESIS#########################
    score = score_model(H_theta, y_sub_test)
    print(f'accuracy = {np.around(score, decimals=3)}')
    ############################################################

    ######APPLTYING THE MODEL FOR OTHER FEATURES################
    X_sepal = iris.data[:, :2]
    y = np.array(iris.target)
    y = np.atleast_2d(y).T

    Thetas_sepal = one_vs_all_classification(X_sepal, y)
    Thetas_sepal = np.array(Thetas_sepal).reshape(3, -1)

    X_sepal = np.c_[np.ones(X_sepal.shape[0]), X_sepal]
    H_theta_sepal = np.dot(X_sepal, Thetas_sepal)

    score_sepal = score_model(H_theta_sepal, y)
    print(f'accuracy = {np.around(score_sepal, decimals=3)}')


if __name__ == '__main__':
    main()
