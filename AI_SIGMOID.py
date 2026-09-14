import numpy as np

LEARNING_RATE = 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inverse(y):
    return np.log(y / (1 - y))


class Layer:
    def __init__(self, neurons, inputs):
        self.matrix = np.random.rand(neurons, inputs) * np.sqrt(2 / inputs)
        self.bias = np.random.rand(neurons)
        self.neuron_activations = np.zeros(neurons)

        self.neuron_gradients = np.zeros(neurons)
        self.weight_gradients = np.zeros_like(self.matrix)

    def calculate(self, inputs):
        results = sigmoid(self.matrix @ inputs + self.bias)
        self.neuron_activations = results
        return results

    def learn(self):
        self.matrix -= LEARNING_RATE * self.weight_gradients
        self.bias -= LEARNING_RATE * self.neuron_gradients

    def find_gradient(self):
        pass


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
        for layer in self.layers:
            results = layer.calculate(results)
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
                    layer.neuron_gradients[g] = da * a * (1 - a)
            else:
                for w, neuron in enumerate(layer.matrix):
                    a = layer.neuron_activations[w]

                    next_layer = self.layers[nh + 1]

                    da = np.sum(
                        next_layer.neuron_gradients * next_layer.matrix[:, w]  #
                    )

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
information1 = sigmoid(0.45)



a = AI(2, [5, 4, 3, 2, 1])
for x in range(200):
    print(sigmoid_inverse(a.calculate(i)))
    a.learn(np.array([information]))
    print(sigmoid_inverse(a.calculate(i1)))
    a.learn(np.array([information1]))
    
print(sigmoid_inverse(a.calculate(i1)))
print(sigmoid_inverse(a.calculate(i)))