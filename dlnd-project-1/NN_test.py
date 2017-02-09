"""
Author: Walter
Date: 09/02/17
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from benchmark import MSE#, NeuralNetwork
from sys import argv
"""
here we built a 2 hidden layer based neural network, the network structure looks as below:

input ---------->   hidden_layer_0   ------------>  hidden_layer_1   -----------> output_layer
	weights				weights				weights
	i_to_h				h_to_h				h_to_o

"""
#Multi-Hidden layer neural network version 0.1.0
#This NeuralNetwork is only used for regression tasks, if you want use it for classification tasks, you need add activation function to final_outputs
class NeuralNetwork_V1(object):
	#initial all parameters here
	def __init__(self, input_nodes, hidden_nodes_0,hidden_nodes_1, output_nodes, learning_rate):
		#Set number of nodes in input, hidden and output layers.
		self.input_nodes = input_nodes
		self.hidden_nodes_0 = hidden_nodes_0
		self.hidden_nodes_1 = hidden_nodes_1
		self.output_nodes = output_nodes
		#Initialize all weights
		self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes_0**-0.5, (self.hidden_nodes_0, self.input_nodes))
		self.weights_hidden_to_hidden = np.random.normal(0.0, self.hidden_nodes_1**-0.5,(self.hidden_nodes_1, self.hidden_nodes_0))
		self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, (self.output_nodes, self.hidden_nodes_1))
		self.lr = learning_rate
		#activation function
		self.activation = self.sigmoid

	#sigmoid function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	#code for training tasks
	def train(self, inputs_list, label_list):
		self.passforward(inputs_list)
		#Backward through
		#output errors(label_list - label_list)
		error = label_list - self.final_outputs
		#Backpropagated error
		#errors propagated to the hidden layer_1
		hhidden_errors = np.dot(self.weights_hidden_to_output.T, error) * self.hidden_outputs_1 * (1 - self.hidden_outputs_1)
		#hidden layer gradients for the hidden layer_1
		hhidden_grad = np.dot(error, self.hidden_outputs_1.T)
		#hidden propagated to the hidden layer_0
		hidden_errors = np.dot(self.weights_hidden_to_hidden.T, hhidden_errors) * self.hidden_outputs * (1 - self.hidden_outputs)
		#hidden layer gradients for the hidden layer_0
		hiden_grad = np.dot(hhidden_errors, self.hidden_outputs.T)
		#update weights
		#update hidden-to-output weights with gradient descent step
		self.weights_hidden_to_output += self.lr * hhidden_grad / self.inputs.shape[1]
		#update hidden-to-hidden weights with gradient descent step
		self.weights_hidden_to_hidden += self.lr * hiden_grad/ self.inputs.shape[1]
		#update input-to-hidden weights with gradient descent step
		self.weights_input_to_hidden += self.lr * np.dot(hidden_errors, self.inputs.T)/self.inputs.shape[1]

	#code for predict tasks
	def run(self, inputs_list):
		self.passforward(inputs_list)
		return self.final_outputs

	#passforward
	def passforward(self, inputs_list):
		#Run a forward pass through the Network
		self.inputs = np.array(inputs_list, ndmin=2).T
		#Hidden Layer
		#signals into hidden layer
		self.hidden_inputs = np.dot(self.weights_input_to_hidden, self.inputs)
		#signals from hidden layer
		self.hidden_outputs = self.activation(self.hidden_inputs)
		#signals from hidden layer to hidden layer
		self.hidden_inputs_1 = np.dot(self.weights_hidden_to_hidden, self.hidden_outputs)
		#sinals from hidden to hidden layer
		self.hidden_outputs_1 = self.activation(self.hidden_inputs_1)
		#Output Layer
		#signals into final output layer
		self.final_inputs = np.dot(self.weights_hidden_to_output, self.hidden_outputs_1)
		#signals from final output layer
		self.final_outputs = self.final_inputs
def MSE(y, Y):
    return np.mean((y-Y)**2)

#using pandas dummy operation for dummy_fields and return the data
def dummy_variables(rides):
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    print("After dummies, the data looks as {0}".format(data.head()))
    return data

#scaling the input data
def scaling(data):
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std
    return data, scaled_features

def train_test_split(data):
    # Save the last 21 days
    test_data = data[-21*24:]
    data = data[:-21*24]
    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    return features, targets, test_features, test_targets, test_data

def read_data(Display = False):
    data_path = 'bike-sharing-dataset/hour.csv'
    rides = pd.read_csv(data_path)
    print("first 5 data samples as {0}".format(rides.head()))
    #plot the dataset by use matplotlib
    if(Display):rides[:24*10].plot(x='dteday', y='cnt')
    data = dummy_variables(rides)
    data, scaled_features = scaling(data)
    features, targets, test_features, test_targets, test_data = train_test_split(data)
    return features, targets, test_features, test_targets, scaled_features, test_data, rides

#using sgd to train the network and visiulize the train loss & valid loss
def sgd_tarining(learningrate, hidden_nodes_0, hidden_nodes_1, epochs,train_features, train_targets, val_features, val_targets):
    ### Set the hyperparameters here ###
    epochs = int(epochs)
    learning_rate = float(learningrate)
    hidden_nodes_0 = int(hidden_nodes_0)
    hidden_nodes_1 = int(hidden_nodes_1)
    output_nodes = 1
    N_i = train_features.shape[1]
    #network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
    network = NeuralNetwork_V1(N_i, hidden_nodes_0, hidden_nodes_1, output_nodes, learning_rate)
    losses = {'train':[], 'validation':[]}
    for e in range(epochs):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        for record, target in zip(train_features.ix[batch].values,
                                 train_targets.ix[batch]['cnt']):
            #print(record.shape)
            network.train(record, target)

        # Printing out the training progress
        train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
        print("\rProgress: " + str(100 * e/float(epochs))[:4] \
                        + "% ... Training loss: " + str(train_loss)[:5] \
                        + " ... Validation loss: " + str(val_loss)[:5])

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    plt.ylim(ymax=1.0)
    plt.show()
    return network

def run(learningrate, epochs, hidden_nodes_0, hidden_nodes_1):
    #read data from dataset and prepara for the input data
    features, targets, test_features, test_targets, scaled_features, test_data, rides = read_data()
    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]
    network = sgd_tarining(learningrate, hidden_nodes_0, hidden_nodes_1, epochs, train_features, train_targets, val_features, val_targets)

if __name__ == "__main__":
    script, first, second, third, fourth = argv
    print(script, first, second, third, fourth)
    run(first, second, third, fourth)
