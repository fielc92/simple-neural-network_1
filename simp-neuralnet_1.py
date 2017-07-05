#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 05 20:48:59 2017
@author: Friedrich
@title: simple neural network 1
@credits: Milo Spencer-Harper (original author)
('
https://medium.com
/technology-invention-and-more
/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
')
"""
from numpy import exp, array, random, dot


class NeuralNetwork():
    """Single neural network follows."""

    def __init__(self):
        """Thin init.

        Seed the random number generator, so it gnerates the same numbers
        everytime the program runs.
        We model a single neuron, with 3 input connections
        and 1 output connection.
        We assign random weights to a 3 x 1 matrix, with values in the range
        -1 to 1 and mean 0.
        """
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        """The Sigmond function.

        Gives an S-shape curve. We pass the weighted sum of the inputs
        through this function to normalise them between 0 and 1.
        """
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        """The derivative of the Sigmoid function.

        This is the gradient of the Sigmoid curve.
        It indicates how confident we are about the existing weight.
        """
        return x * (1 - x)

    def train(self,
              training_set_inputs,
              training_set_outputs,
              number_of_training_iterations):
        """We train the neural network through a process of trail and error.

        Adjusting the synaptic weights each time.
        1, Pass the training set inputs through our (single) neural network.
        2. Calculate the error (The difference between the desired output
        and the predicted output).
        3. Multiply the error by the input and again by the gradient of the
        Sigmoid curve. This means less confident weights are adjusted more.
        This means inputs, which are zero, do not cause changes to the weights.
        4. Adjust the weights.
        """
        for iteration in range(number_of_training_iterations):
            # 1
            output = self.think(training_set_inputs)
            # 2.
            error = training_set_outputs - output
            # 3.
            adjustment = dot(training_set_inputs.T,
                             error * self.__sigmoid_derivative(output))
            # 4.
            self.synaptic_weights += adjustment

    def think(self, inputs):
        """The neural network thinks.

        Pass inputs through our neural network (our single neuron).
        """
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # Initialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print('\nRandom starting synaptic weights:')
    print(neural_network.synaptic_weights)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1],
                                 [1, 1, 1],
                                 [1, 0, 1],
                                 [0, 1, 1]])

    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10 000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print('\nNew synaptic weights after training:')
    print(neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print('\nConsidering new situation [1, 0, 0] -> ?:')
    print(neural_network.think(array([1, 0, 0])))
