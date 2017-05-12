from random import seed
from random import random
import math

class Neuralnetworks:
  def __init__(self, n_inputs, n_hidden):
    print("Creating the neural network with " + str(n_inputs) + " neurons in the hidden layer")
    self.network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    self.network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]}]
    self.network.append(output_layer)


  # The activation is the sum of its inputs times the weights
  def neuron_activation(self, weights, inputs):
    activation = 0.0
    for i in range(len(weights)):
      activation += weights[i] * inputs[i]
    return activation

  # Activation function implemented as sigmoid
  def activation_funcion(self,activation):
    print ("sigmoid: " + str((1.0 / (1.0 + math.exp(-activation)))))
    return 1.0 / (1.0 + math.exp(-activation))

  # Forward propagation function
  def forward_propagation(self,row):
    inputs = row
    for layer in self.network:
      print("Novo layer")
      new_inputs = []
      inputs.append(1)
      for neuron in layer:
        print("Novo Neuoronio")
        activation = self.neuron_activation(neuron["weights"],inputs)
        neuron["output"] = self.activation_funcion(activation)
        new_inputs.append(neuron['output'])
        print(new_inputs)
      inputs = new_inputs
    return inputs

