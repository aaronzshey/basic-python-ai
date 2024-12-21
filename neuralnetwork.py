import numpy as np

class NeuralNetwork:
    def __init__ (self, learning_rate): 
        # init the weights with a random number
        self.weights = np.array([np.random.randn(), np.random.randn()])

        # init the bias with a random number
        self.bias = np.random.randn()

        # learning rate is user defined
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias 
        return self.sigmoid(layer_1)  
    
    # gradient is synonymous with derivative
    def compute_gradients(self, input_vector, target):
        prediction = self.predict(input_vector) 
        derror_dprediction = 2 * (prediction - target)

        # error calculation for layer 1?
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        dprediction_dlayer1 = self.sigmoid_derivative(layer_1)

        # we just set the bias to 1? this tutorial is really strange
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights) 

        return derror_dbias, derror_dweights

    # update the weights and bias, finally
    def update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # pick a random data point
            random_data_index = np.random.randint(len(input_vectors))   

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # compute and update gradients

            derror_dbias, derror_dweights = self.compute_gradients(
                input_vector, target
            )

            self.update_parameters(derror_dbias, derror_dweights)

            if current_iteration % 100 == 0: 
                cumulative_error = 0

                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

            return cumulative_errors
        







