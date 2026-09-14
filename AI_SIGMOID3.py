import numpy as np

LEARNING_RATE = 0.1
CLIPPING = [-5, 5]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inverse(y):
    return np.log(y / (1 - y))


class Layer:
    def __init__(self, neurons, inputs):
        self.matrix = np.random.randn(neurons, inputs) * np.sqrt(2 / inputs)
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
            self.neuron_activations = sigmoid(self.z)

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
        for h, layer in enumerate(self.layers):
            End = h == len(self.layers) - 1
            results = layer.calculate(results, End)
        self.prediction = results[0]
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
                    layer.neuron_gradients[g] = da
            else:
                next_layer = self.layers[nh + 1]

                for w in range(len(layer.neuron_activations)):
                    a = layer.neuron_activations[w]

                    da = np.sum(next_layer.neuron_gradients * next_layer.matrix[:, w])

                    # sigmoid
                    layer.neuron_gradients[w] = da * a * (1 - a)

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


i = np.array([2, 12])
information = sigmoid(1)
i1 = np.array([8, 4.2])
information1 = sigmoid(0.1)


a = AI(2, [16, 32, 32, 16, 8, 1])
inputs = [np.array([1, 2]), np.array([2, 8]), np.array([14, -15]), np.array([5, 4.41])]

targets = [np.array([0.43]), np.array([0.3]), np.array([0.6]), np.array([2.41])]

for epoch in range(2000):

    for x, y in zip(inputs, targets):
        a.calculate(x)
        a.learn(y)
print(a.calculate(inputs[0]))
print(a.calculate(inputs[1]))
print(a.calculate(inputs[2]))
print(a.calculate(inputs[3]))
