from random import seed
from random import random
import math

class Neuralnetworks:
  def __init__(self, n_inputs, n_hidden, n_outputs):
    print("Creating the neural network with " + str(n_hidden) + " neurons in the hidden layer and " + str(n_outputs) + " outputs")
    self.network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    self.network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
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

  # sigmoid derivate
  def sigmoid_derivate(self,sigmoid):
    return sigmoid* ( 1.0 - sigmoid)

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

        # The output key will hold all values that the neuron activation function resulted until the
        # back propagation process. It should be cleaned after each back propagation
        if "output" in  neuron:
          neuron["output"].append(self.activation_funcion(activation))
        else:
          neuron["output"] = [self.activation_funcion(activation)]

        #neuron["output"] = self.activation_funcion(activation)
        #new_inputs.append(neuron['output'])

        new_inputs.append(neuron['output'][-1])
        print(new_inputs)
      inputs = new_inputs
    return inputs


  # Loss function implemented as cross entropy
  def loss_function(self,output,index_correct_class):
    loss = 0.0
    for i in range(len(output)):
      # verify if the current i is the correct class and set y_k accordingly
      if i == index_correct_class:
        y_k = 1
      else:
        y_k = 0
      # add the cross entropy of this class
      loss +=  - y_k * math.log10(output[i]) - (1 - y_k) * math.log10(1- output[i])
    return loss

  # auxiliar function to break a vector into num sub-vectors of same size
  # size 10 => num  500
  # size 50 => num  100
  def chunk_it(self,data,num):
    avg = len(data) / float(num)
    sub_vetors = []
    last = 0.0
    while last < len(data):
      sub_vetors.append(data[int(last):int(last + avg)])
      last += avg
    return sub_vetors

  # Back propagation algorithm regarding the GD style: full generation, stochastic and mini-batch
  def back_propagate_error(self,batch_sized_data, batch_sized_classes):

    #calculate the accumulated loss function result for each one of the entries from the batch_sized_data
    # and store the neurons output of each one inside the network
    loss = 0.0

    for i in range(len(batch_sized_data)):
      loss += self.loss_function(self.forward_propagation(batch_sized_data[i]),batch_sized_classes[i])

    # The GD  loss should be calculated as the mean of the losses
    loss = loss / len(batch_sized_data)

    # Now that we have the loss, we should update the neuron's weights
    # In order to do that, we should iterate from the deepest layer to the shallowest
    for i in  reversed(range(len(self.network))):
      layer = self.network[i]
      errors = list()
      # check if it's the deepest layer, because if it is, the error is calculated with out
      # the gradient
      if i != len(self.network)-1:
        for j in range(len(layer)):
          error = 0.0
          for neuron in self.network[i + 1]:

            # Here we will use the error as the mean of the batch
            for m in range(len(neuron['delta'])):
              error += (neuron['weights'][j] * neuron['delta'][m])
            error = error / len(batch_sized_data)
            #error += (neuron['weights'][j] * neuron['delta'])
            errors.append(error)
      else:
        for j in range(len(layer)):
          neuron = layer[j]
          # The error of the deepest layer is the loss calculated before
          errors.append(loss)

      for j in range(len(layer)):
        neuron = layer[j]
        neuron['delta'] = []
        for k in range(len(batch_sized_data)):
          neuron['delta'].append(errors[j] * self.sigmoid_derivate(neuron['output'][k]))





