import numpy as np
import time

LEARNING_RATE = 0.01


def relu(x):  # leaky relu
    return np.where(x > 0, x, 0.01 * x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)


class Layer:
    def __init__(self, neurons, inputs):
        self.matrix = np.random.randn(neurons, inputs) * np.sqrt(2 / inputs)
        self.bias = np.zeros(neurons)

        self.z = np.zeros(neurons)
        self.neuron_activations = np.zeros(neurons)

        self.neuron_gradients = np.zeros(neurons)
        self.weight_gradients = np.zeros_like(self.matrix)

    def calculate(self, inputs, End = False):
        self.z = self.matrix @ inputs + self.bias
        if End:
            self.neuron_activations = self.z
        else:
            self.neuron_activations = relu(self.z)

        return self.neuron_activations

    def learn(self):
        self.matrix -= LEARNING_RATE * self.weight_gradients
        self.bias -= LEARNING_RATE * self.neuron_gradients


class AI:
    def __init__(self, inputs, pattern=None):
        if pattern is None:
            pattern = [3, 3, 3, 1]

        self.layers = []

        self.layers.append(Layer(pattern[0], inputs))
        for h in range(1, len(pattern)):
            self.layers.append(Layer(pattern[h], pattern[h - 1]))
        self.prediction = None
        self.inputs = None

    def calculate(self, inputs):
        self.inputs = inputs
        results = inputs
        for i, layer in enumerate(self.layers):
            End = (i == len(self.layers) - 1)
            results = layer.calculate(results, End)
        self.prediction = results
        return results

    def learn(self, information):
        # every last neuron should know what he should say. number of last neurons = number of information

        for h, layer in enumerate(self.layers[::-1]):
            nh = len(self.layers) - (1 + h)

            da = 0
            if h == 0:
                for g in range(len(layer.neuron_activations)):
                    a = layer.neuron_activations[g]
                    da = a - information[g]
                    layer.neuron_gradients[g] = da * relu_derivative(layer.z[g])
            else:
                for w, neuron in enumerate(layer.matrix):
                    a = layer.neuron_activations[w]

                    next_layer = self.layers[nh + 1]

                    da = np.sum(
                        next_layer.neuron_gradients * next_layer.matrix[:, w]  #
                    )

                    layer.neuron_gradients[w] = da * relu_derivative(layer.z[w])

        for h, layer in enumerate(self.layers[::-1]):
            nh = len(self.layers) - (1 + h)
            if nh != 0:
                previous_layer = self.layers[nh - 1]
                layer.weight_gradients = np.outer(
                    layer.neuron_gradients, previous_layer.neuron_activations
                )
                layer.learn()

            else:
                layer.weight_gradients = np.outer(layer.neuron_gradients, self.inputs)
                layer.learn()


i = np.array([1, 2])
information = 1.231

i2 = np.array([2, 8])
information2 = 0.3


a = AI(2, [2, 5, 4, 3, 2, 2, 1])
for x in range(2000):
    # print()
    a.calculate(i)
    a.learn(np.array([information]))
    # print()
    a.calculate(i2)
    a.learn(np.array([information2]))


print(a.calculate(i))
print(a.calculate(i2))
print()
print(a.calculate(np.array([1, 2])))
print(a.calculate(np.array([2, 8])))
print(a.calculate(np.array([1, 8])))
print(a.calculate(np.array([2, 2])))
