import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_data(path):
    '''
    Load data using pandas
    :param path: path to cvs file
    :return: data frame of the given data
    '''
    return pd.read_csv(path)


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



def cost_function(X, y, w):
    '''
    The objective of linear regression is to minimize the cost function
    J(w) = 1/2m Σ (h(x_i) - y_i)^2
    Where = h(x) = w.T*x = w_0 + w_1*x_1
    :param X: X: Input vector
    :param y: y: true labels
    :param w: w: weights
    :return: results of the cost function
    '''
    m = len(y)
    y_pred = X.dot(w) # w.T*x
    error = (y_pred - y) ** 2 # (h(x_i) - y_i)^2
    cost = 1/(2*m) * np.sum(error) # 1/2m Σ (h(x_i) - y_i)^2
    return cost


def train(X, y, w, num_of_itr, lr=0.01):
    '''
    GRADIENT DESCENT:
    Minimize the cost function by updating the equation until convergence
    w := w - 1/m Σ (h(x_i) - y_i)*x
    Where = h(x) = w.T*x = w_0 + w_1*x_1
    :param X: input vector
    :param y: true labels
    :param w: weights
    :param lr: learning rate
    :param num_of_itr: number of iterations
    :return: weights, costs history
    '''
    m = len(y)
    # costs history
    costs = []
    # update the weights for (number of iterations) times
    for i in range(num_of_itr):
        # calculate: h(x) = w.T*x
        y_pred = X.dot(w)
        # calculate: h(x) = w.T*x = w_0 + w_1*x_1
        error = np.dot(X.transpose(), (y_pred - y))
        # calculate: w := w - 1/m Σ (h(x_i) - y_i)*x
        w -= lr * 1/m * error # w := w - 1/m Σ (h(x_i) - y_i)*x
        # save current cost function
        costs.append(cost_function(X, y, w))
    return w, costs


def predict(x, w):
    '''
    # Calculate the prediction - h(x) = w.T*x
    :param x: input vector
    :param w: weights
    :return: prediction that calculateed by w.T*x
    '''
    return np.dot(w.transpose(), x)


def plot_convergence(costs):
    '''
    Plotting the convergence
    :param costs: history of costs
    :return: None, show the plot
    '''
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('W')

    plt.title('Convergence')
    plt.show()





