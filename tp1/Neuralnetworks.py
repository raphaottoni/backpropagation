from random import seed
#from random import random
from numpy import random
import math

class Neuralnetworks:

  seed(901)
  def __init__(self, n_inputs, n_hidden, n_outputs):
    print("Creating the neural network with " + str(n_hidden) + " neurons in the hidden layer and " + str(n_outputs) + " outputs")
    self.network = list()
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs= n_outputs
    hidden_layer = [{'weights':[random.random()/math.sqrt(n_hidden) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    self.network.append(hidden_layer)
    output_layer = [{'weights':[random.random()/math.sqrt(n_hidden) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    self.network.append(output_layer)


  # The activation is the sum of its inputs times the weights
  def neuron_activation(self, weights, inputs):
    activation = 0.0
    for i in range(len(weights)):
      activation += weights[i] * inputs[i]
    return activation

  # Activation function implemented as sigmoid
  def activation_funcion(self,activation):
    return  1.0 / (1.0 + math.exp(-activation))

  # sigmoid derivate
  def sigmoid_derivate(self,sigmoid):
    return sigmoid* ( 1.0 - sigmoid)

  # Forward propagation function
  def forward_propagation(self,row):
    inputs = row
    for layer in self.network:
      #print("Novo layer")
      new_inputs = []
      inputs.append(1)
      for neuron in layer:
        #print("Novo Neuoronio")
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
        #print(neuron['output'])
        #print(neuron['output'][-1])
      inputs = new_inputs
    #print(inputs)
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
      loss +=  - y_k * math.log(output[i]) - (1 - y_k) * math.log(1 - output[i])
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
      #print(self.forward_propagation(batch_sized_data[i]))
      loss += self.loss_function(self.forward_propagation(batch_sized_data[i]),batch_sized_classes[i])

    # The GD  loss should be calculated as the mean of the losses
    #loss = loss / len(batch_sized_data)

    # Print the loss of this mini-batch
    #print("The loss of this iteration was: " + str(loss))

    # Now that we have the loss, we should update the neuron's weights
    # In order to do that, we should iterate from the deepest layer to the shallowest
    for i in  reversed(range(len(self.network))):
      layer = self.network[i]
      #errors = list()
      errors = {}
      # check if it's the deepest layer, because if it is, the error is calculated with out
      # the gradient
      if i != len(self.network)-1:
        for j in range(len(layer)):
          #error = 0.0
          errors[j]= []
          for m in range(len(layer[j]['output'])):
            erro= 0.0
            for neuron in self.network[i+1]:
              #print("Colocando erro["+ str(j)+ "]: " + str( neuron['weights'][j]) + " * "+ str(neuron['delta'][m]) + " = " + str(neuron['weights'][j] * neuron['delta'][m]))
              erro += neuron['weights'][j] * neuron['delta'][m]
            errors[j].append(erro)

          #for neuron in self.network[i + 1]:
          #  #print("Olhando o tamanho de delta")
          #  #print(neuron['delta'])
          #  #print(neuron['weights'])
          #  #print(layer)
          #  #print(len(layer))
          #  #for m in range(len(neuron['delta'])):
          #  error = 0.0
          #  for m in range(len(neuron['delta'])):
          #    #error += (neuron['weights'][j] * neuron['delta'][m])
          #    error += (neuron['weights'][j] * neuron['delta'][m])

          ##errors[j].append(neuron['weights'][j] * neuron['delta'][m])
          #errors[j].append(error)

            #error = error / len(batch_sized_data)
            #error += (neuron['weights'][j] * neuron['delta'])
            #errors.append(error)
        #print("Errors primeira camada")
        #print(errors)
      else:
        for j in range(len(layer)):
          neuron = layer[j]
          # The error of each neuron of the output layer is the diference from the expected class
          erro = 0.0
          errors[j]= []
          for m in range(len(neuron['output'])):
            #partial +=  neuron['output'][m] - batch_sized_classes[m]
            #erro +=   batch_sized_classes[m] - neuron['output'][m]
            #errors[j].append(batch_sized_classes[m] - neuron['output'][m])
            if j == batch_sized_classes[m]:
              target = 1.0
            else:
              target = 0.0
            #print("O Target :  " + str(target))
            errors[j].append(target - neuron['output'][m])
            #errors[j].append(neuron['output'][m]-target)
            #errors[j].append(neuron['output'][m] - batch_sized_classes[m])
            #print(str(neuron['output'][m])  + " - " + str(batch_sized_classes[m]))
            #print(str(batch_sized_classes[m])  + " - " + str(neuron['output'][m]))
          #errors.append(erro)
        #print("Errors ultima camada")
        #print(errors)
      for j in range(len(layer)):
        neuron = layer[j]
        neuron['delta'] = []
        #print(neuron)
        for k in range(len(batch_sized_data)):
          #print(self.sigmoid_derivate(neuron['output'][k]))
          #print(neuron['output'][k])
          neuron['delta'].append(errors[j][k] * self.sigmoid_derivate(neuron['output'][k]))
          #print("Delta: " + str(errors[j][k]) + " * "  +str( self.sigmoid_derivate(neuron['output'][k])))
          #print(neuron)

    return loss


  # Update network weights with error
  def update_weights(self,batch_sized_data,l_rate):
    for i in range(len(self.network)):

      # Finds the inputs of each layer
      inputs = batch_sized_data
      if i != 0:
        inputs_temp = {k: [] for k in range(len(self.network[i-1][0]['output']))}
        for neuron in self.network[i -1]:
          for output in range(len(neuron['output'])):
            inputs_temp[output].append(neuron['output'][output])

        inputs = []
        for v in range(len(inputs_temp)):
          inputs_temp[v].append(1)
          inputs.append(inputs_temp[v])

      #print("----> " + str(inputs))
      #print(len(inputs))
      #for each neuron of this layer
      for neuron in self.network[i]:
        partial = [0.0] * len(neuron['weights'])
        #print(len(inputs))
        #partial = {}
        for j in range(len(inputs)):
          #print(len(inputs[j]))
          for m in range(len(inputs[j])):
              #print("adicionando: "  + str(neuron['delta'][j]) + " * " + str(inputs[j][m]) +  " em Partial["+str(m)+"]")
              #neuron["weights"][m] -= neuron['delta'][j] * inputs[j][m]
              #partial[m] += neuron['delta'][j] * inputs[j][m] #* l_rate * 1.0/len(inputs)
              neuron["weights"][m] += neuron['delta'][j] * inputs[j][m] * l_rate * 1.0/len(inputs)
        #print("calculo: " + str(l_rate) + " * " + str(neuron['delta'][j]) + " * " + str(inputs[j][m]) + " * " + str(1/len(inputs)))
        #for v in range(len(neuron["weights"])):
          #print("atualizando ["+str(v)+"]: " + str(neuron["weights"][v]) + " + (" + str(partial[v]) +" * "+  str(l_rate) + " * " +str(1.0/len(inputs)))
          #neuron["weights"][v] = neuron["weights"][v]  + (partial[v] * l_rate * 1.0/len(inputs))
          #neuron["weights"][v] = neuron["weights"][v]  + partial[v]
          #print("= " +str( neuron["weights"][v] ))


    # clean the outputs for the next back_propagation_error
    for i in range(len(self.network)):
      for neuron in self.network[i]:
        neuron.pop('output', None)

  # General function to train the network
  def train(self,subvector_size,data,data_classes,l_rate,n_epoch):
    #subdivide the data according to GD method
    batch_sized_data = self.chunk_it(data,subvector_size)
    batch_sized_classes = self.chunk_it(data_classes,subvector_size)
    #for each epoch
    for i in range(n_epoch):
      loss = 0.0

      #for each mini-batch
      for j in range(len(batch_sized_data)):
        #for i in range(len(batch_sized_data)):
        #  loss += self.loss_function(self.forward_propagation(batch_sized_data[i]),batch_sized_classes[i])
        loss += self.back_propagate_error(batch_sized_data[j],batch_sized_classes[j])
        #temp = self.back_propagate_error(batch_sized_data[j],batch_sized_classes[j])

        #print("retornei :" +str(loss))
        # remove extra bias added after each epoch
        for entry in range(len(batch_sized_data[j])):
          #print(self.network)
          if len(batch_sized_data[j][entry]) > (self.n_inputs + 1):
            batch_sized_data[j][entry].pop()
        #print(self.network)
        self.update_weights(batch_sized_data[j],l_rate)
        #print(self.network)

      # shuffle the data
      c = list(zip(data,data_classes))
      random.shuffle(c)
      data,data_classes = zip(*c)
      #random.shuffle(data)
      print(str(i+1) + ", " + str(loss/len(data)))
