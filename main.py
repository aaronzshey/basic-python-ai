# code adapted from https://realpython.com/python-ai-neural-network/
import neuralnetwork as NeuralNetwork
import matplotlib.pyplot as plt 
import numpy as np


input_vectors = np.array(
        [
            [3, 1.5],
            [2, 1],
            [4, 1.5],
            [3, 4],
            [3.5, 0.5],
            [2, 0.5],
            [5.5, 1],
            [1, 1],
        ]
    )

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])

learning_rate = 0.1

neural_network = NeuralNetwork.NeuralNetwork(learning_rate)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")