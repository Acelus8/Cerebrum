import numpy as np
import time



def relu(x):  # leaky relu
    return np.where(x > 0, x, 0.01 * x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)






class Layer:
    def __init__(self, neurons, inputs, RW = 1, LR = 0.003, CLIPPING = [-100, 100], algorythm = "Learn"):
        if algorythm == "Learn":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0
        
            self.clipping = CLIPPING
            self.learning_rate = LR
        
        
        
        self.matrix = (
            np.random.randn(neurons, inputs) * np.sqrt(2 / inputs) * RW
        )
        self.bias = np.zeros(neurons)


        
        if algorythm == "Learn":
            self.z = np.zeros(neurons)
            self.neuron_activations = np.zeros(neurons)
            self.neuron_gradients = np.zeros(neurons)
            self.weight_gradients = np.zeros_like(self.matrix)
           
            self.m_weights = np.zeros_like(self.matrix)
            self.v_weights = np.zeros_like(self.matrix)
        

    def calculate(self, inputs, End=False):
        self.z = self.matrix @ inputs + self.bias
        if End:
            self.neuron_activations = self.z
        else:
            self.neuron_activations = relu(self.z)

        return self.neuron_activations

    def learn(self):
        
        self.neuron_gradients = np.clip(self.neuron_gradients, *self.clipping)
        self.weight_gradients = np.clip(self.weight_gradients, *self.clipping)
        
        self.t += 1

        self.m_weights = (self.beta1 * self.m_weights + (1 - self.beta1) * self.weight_gradients)
        self.v_weights = (self.beta2 * self.v_weights + (1 - self.beta2) * self.weight_gradients ** 2)
        
        m_hat = self.m_weights / (1 - self.beta1 ** self.t)
        v_hat = self.v_weights / (1 - self.beta2 ** self.t)


        self.matrix -= (self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))               #ADAM
        self.bias -= self.learning_rate * self.neuron_gradients
        
    def mutate(self):
        pass


class AI:
    def __init__(self, inputs, pattern=None, al="Learn", open=None):

        if pattern is None:
            pattern = [3, 3, 3, 1]

        self.pattern = pattern
        self.inputs = inputs
        self.prediction = None

        self.layers = []

        self.layers.append(Layer(pattern[0], inputs, algorythm=al))

        for h in range(1, len(pattern)):
            self.layers.append(
                Layer(pattern[h], pattern[h - 1], algorythm=al)
            )

        if open:
            self.load(open)

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


    def save(self, name):
        arrays = {}

        arrays["pattern"] = np.array(self.pattern)

        for i, layer in enumerate(self.layers):
            arrays[f"w{i}"] = layer.matrix
            arrays[f"b{i}"] = layer.bias
            # Adam
            arrays[f"m{i}"] = layer.m_weights
            arrays[f"v{i}"] = layer.v_weights
            arrays[f"t{i}"] = layer.t

        np.savez(name, **arrays)


    def load(self, name):
        data = np.load(name)

        self.pattern = data["pattern"].tolist()
        self.layers = []
        self.layers.append(
            Layer(self.pattern[0], self.inputs)
        )

        for i in range(1, len(self.pattern)):
            self.layers.append(
                Layer(self.pattern[i], self.pattern[i - 1])
            )

        for i, layer in enumerate(self.layers):
            layer.matrix = data[f"w{i}"]
            layer.bias = data[f"b{i}"]

            # Adam
            layer.m_weights = data[f"m{i}"]
            layer.v_weights = data[f"v{i}"]
            layer.t = int(data[f"t{i}"]) 





ai = AI(2, [8, 8, 8, 8, 8, 8, 1])

data = [
    ([10, 0], 0),
    ([12, 0.89], 6.43),
    ([14, 2.12], 12.86),
    ([15, 3.83], 19.29),
    ([16, 5.98], 25.71),
    ([17, 8.75], 32.14),
    ([18, 12.23], 38.57),
    ([19, 16.56], 45),
    ([13.47, 4.26], 27.34),
    ([17.83, 3.71], 31.82)
]

for epoch in range(10000):
    for inputs, target in data:
        ai.calculate(inputs)
        ai.learn([target])

    if epoch % 1000 == 0:
        print("epoch:", epoch)

print("\n--- TEST ---")

for inputs, target in data:
    prediction = ai.calculate(inputs)

    print(
        "input:", inputs,
        "target:", target,
        "prediction:", prediction
    )