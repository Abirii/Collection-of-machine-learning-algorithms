import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.01, num_of_iter=10000):
        '''
        Initialize the Logistic Regression with
        weights, bias and:
        :param lr: learning rate
        :param num_of_iter: number of iteration
        '''
        self.lr = lr
        self.num_of_iter = num_of_iter
        self.weights = None


    def sigmoid(self, z):
        '''
        calc sigmoid function
        :param z:
        :return: result of sigmoid function
        '''

        sigmoid = 1/(1 + np.exp(-z))

        return sigmoid


    def fit(self, X, y):
        '''
        Train function (gradient descent)
        :param X: train set (numpy nd array m x n where m is the number of samples and n is the number of features)
        :param y: lebels (m x 1 vector)
        :return:
        '''

        # initialize
        num_of_samples, num_of_features = X.shape
        self.weights = np.zeros(num_of_features)

        # gradient descent
        for i in range(self.num_of_iter):
            z = np.dot(X, self.weights)
            y_pred = self.sigmoid(z)
            dw = (1 / num_of_samples) * np.dot(X.T, (y_pred - y))
            # update rule
            self.weights -= self.lr * dw

    def predict(self, X):
        '''
        prediction function
        :param X: input vector
        :return: prediction
        '''
        z = np.dot(X, self.weights)
        y_pred = self.sigmoid(z)
        print(y_pred)
        if y_pred > 0.5:
            return 1
        return 0




res = LogisticRegression()
X = np.array([[1, 0], [2, 3], [4, 5]])
res.fit(X, [0, 1, 1])

test = np.array([[4, 5]])
print(res.predict(test))





