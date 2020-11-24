import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(path):
    '''
    Load data using pandas
    :param path: path to cvs file
    :return: data frame of the given data
    '''
    return pd.read_csv(path)


def kernel(point, X, tau):
    '''
    The weights depend on the particular point at which we are trying to evaluate the regression
    If |x(i)−x| is small then the weight w(i) is close to 1
    If |x(i)−x| is large then w(i) is small
    Hence θ is chosen by giving a higher "weight" to the errors on training examples close to the query point  x

    Calculate the kernel that define as: w(x, x0) = exp(- (x-x0)^2) / 2τ^2)
    If x is a vector, then this generalizes to be:  w(i)=exp(− (xi−x).T * (xi−x) / 2τ^2)

    :param point: query point  x
    :param X: data
    :param tau: controls how quickly the weight of a training example falls off
    :return: weights eye matrix - the weights of the ith sample store in weights[i,i] (index)
    '''

    # number of samples
    num_of_samples, _ = np.shape(X)

    # initialize eye matrix, the weights of the ith sample store in weights[i,i] (index)
    weights = np.mat(np.eye((num_of_samples)))

    # run over all the samples
    for i in range(num_of_samples):
        # calculate the: xi−x ( ith sample)
        diff = X[i] - point
        # calculate the: w(i)=exp(− (xi−x).T * (xi−x) / 2τ^2), the weights that calculated store in weights[i,i] (index)
        weights[i, i] = np.exp(diff * diff.T / (-2.0 * tau**2))

    return weights


def local_weight(point, X, y, tau):
    '''

    Find the value of θ which minimizes J(θ) , we can differentiate J(θ)  with respect to θ.
    DifferentiateJ(θ) = X.T * W * X * θ - X.T * W * Y
    To find the value of θ which minimizes J(θ), we set DifferentiateJ(θ) = 0
    The weighted least squares solution: θ =  (X.T * W * X)^-1 * X.T *W * Y

    :param point: query point  x
    :param X: data
    :param y: label vector
    :param tau: controls how quickly the weight of a training example falls off
    :return: theta (θ ) which minimizes J(θ)
    '''

    # assign the weight to the given point
    w = kernel(point, X, tau)

    # the local weight: θ =  (X.T * W * X)^-1 * X.T *W * Y
    theta = (X.T * (w*X)).I * (X.T * w * y.T)
    return theta



def localWeightRegression(X, y, tau, show_proc=False):
    '''
    Local Weight Regression.

    :param X: data
    :param y: label vector
    :param tau: controls how quickly the weight of a training example falls off
    :param: show_proc: show h(xi) calculates
    :return: predictions vector
    '''

    # number of samples
    num_of_samples, _ = np.shape(X)

    # initialize the pred vector
    y_pred = np.zeros(num_of_samples)

    # run over all the x points
    for i in range(num_of_samples):
        # pred for given point -> multiplying the local weight of that point with X[i]
        y_pred[i] = X[i] * local_weight(X[i], X, y, tau) # localWeight = theta

        if show_proc == True:
            print(f"h(θ)= {local_weight(X[i], X, y, tau)[0]} + {local_weight(X[i], X, y, tau)[1]}")

    return y_pred



def visualize_data(data, plt_title, x_title, y_title):
    '''
    Visualize the given data
    :param data: data as data frame
    :param plt_title: title of the plot
    :param x_title: title along x axis
    :param y_title: title along y axis
    :return: None, show the plot
    '''
    ax = sns.scatterplot(x=x_title, y=y_title, data=data)
    ax.set_title(plt_title)
    plt.show(ax)




