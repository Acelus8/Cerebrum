import numpy as np
import time

# LEARNING_RATE = 0.005
LEARNING_RATE = 0.003
RANDOM_WEIGHTS = 1
CLIPPING = [-10, 10]


def relu(x):  # leaky relu
    return np.where(x > 0, x, 0.01 * x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)


class Layer:
    def __init__(self, neurons, inputs):
        self.matrix = (
            np.random.randn(neurons, inputs) * np.sqrt(2 / inputs) * RANDOM_WEIGHTS
        )
        self.bias = np.zeros(neurons)

        self.z = np.zeros(neurons)
        self.neuron_activations = np.zeros(neurons)

        self.neuron_gradients = np.zeros(neurons)
        self.weight_gradients = np.zeros_like(self.matrix)

    def calculate(self, inputs, End=False):
        self.z = self.matrix @ inputs + self.bias
        if End:
            self.neuron_activations = self.z
        else:
            self.neuron_activations = relu(self.z)

        return self.neuron_activations

    def learn(self):
        self.neuron_gradients = np.clip(self.neuron_gradients, *CLIPPING)

        self.weight_gradients = np.clip(self.weight_gradients, *CLIPPING)

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
            End = i == len(self.layers) - 1
            results = layer.calculate(results, End)
        self.prediction = results
        return results

    def learn(self, information):
        for h, layer in enumerate(self.layers[::-1]):
            nh = len(self.layers) - (1 + h)
            if h == 0:

                for g in range(len(layer.neuron_activations)):

                    a = layer.neuron_activations[g]
                    da = a - information[g]
                    layer.neuron_gradients[g] = da
            else:

                next_layer = self.layers[nh + 1]
                for w in range(len(layer.neuron_activations)):
                    da = np.sum(next_layer.neuron_gradients * next_layer.matrix[:, w])

                    # Leaky ReLU
                    layer.neuron_gradients[w] = da * relu_derivative(layer.z[w])

        for h, layer in enumerate(self.layers[::-1]):
            nh = len(self.layers) - (1 + h)
            if nh != 0:
                previous_layer = self.layers[nh - 1]
                layer.weight_gradients = np.outer(
                    layer.neuron_gradients, previous_layer.neuron_activations
                )
            else:

                layer.weight_gradients = np.outer(layer.neuron_gradients, self.inputs)

            layer.learn()


i = np.array([1, 2])
information = 1.231

i2 = np.array([2, 8])
information2 = 0.3

i2 = np.array([2, 8])
information2 = 0.3

i2 = np.array([2, 8])
information2 = 0.3


a = AI(2, [8,8,8,8,8,8,1])
inputs = [
    np.array([10.00, 0.00]),
    np.array([12.00, 0.89]),
    np.array([14.00, 2.12]),
    np.array([15.00, 3.83]),
    np.array([16.00, 5.98]),
    np.array([17.00, 8.75]),
    np.array([18.00, 12.23]),
    np.array([19.00, 16.56]),
    # випадкові
    np.array([13.47, 4.26]),
    np.array([17.83, 3.71]),
]

targets = [
    np.array([0.00]),
    np.array([6.43]),
    np.array([12.86]),
    np.array([19.29]),
    np.array([25.71]),
    np.array([32.14]),
    np.array([38.57]),
    np.array([45.00]),
    # випадкові
    np.array([27.34]),
    np.array([31.82]),
]

for epoch in range(20000):

    for x, y in zip(inputs, targets):
        print(a.calculate(x))
        a.learn(y)
    print("----")

    if epoch / 100 != float:
        print("epoch-------------")
    else:
        print()


input("get ready:")
print()
print(a.calculate(np.array([20, 10])))
print(a.calculate(np.array([1, 1])))
print(a.calculate(np.array([15, 0])))
print()
print(47.09)
print(4.15)
print(22.02)

input()
# np.array([47.09])
# np.array(4.15)
# np.array([22.02])
