import numpy as np


class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        '''
        :param no_of_inputs: is used to determine how many weights we need to learn
        :param threshold: is the number of epochs we’ll allow our learning algorithm to iterate through before ending, and it’s defaulted to 100
        :param learning_rate: is used to determine the magnitude of change for our weights during each step through our training data, and is defaulted to 0.01.
        '''
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)



    def predict(self, inputs):
        '''
        This method will house the f(x) = 1 if w · x + b > 0 : 0 otherwise algorithm.
        :param inputs: expects to be an numpy array/vector of a dimension equal to the no_of_inputs parameter
        :return: activiation results
        '''
        # we can take the dot product of the inputs and the self.weights vector with the the first value “removed”,
        # and then add the first value of the self.weights vector to the dot product
        summation = np.dot(inputs, self.weights[1:]) +self.weights[0]

        if summation > 0:
            activiation = 1
        else:
            activiation = 0

        return activiation



    def train(self, training_inputs, labels):
        '''
        :param training_inputs:is expected to be a list made up of numpy vectors to be used as inputs by the predict method.
        :param labels: is expected to be a numpy array of expected output values for each of the corresponding inputs in the training_inputs list.
        :return:
        '''

        # run a number of times equal to the threshold argument we passed into the Perceptron constructor
        for _ in range(self.threshold):

            # We zip training_inputs and labels together to create a new iterable object
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # find the error, label — prediction, then we multiply it by our self.learning_rate, and by our inputs vector
                self.weights[1:] = self.learning_rate * (label - prediction) * inputs
                # We update the bias in the same way as the other weights, except, we don’t multiply it by the inputs vector.
                self.weights[0] += self.learning_rate * (label - prediction)








