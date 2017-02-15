import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv
from miniflow import *
from sklearn.utils import shuffle, resample

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

def nn(X_, y_, V_X_, V_y_ ,hidden_nodes_0, hidden_nodes_1, learnrate, epochs):
    y_ = np.array(y_['cnt'])
    V_y_ = np.array(V_y_['cnt'])
    #print(X_.shape, y_.shape)
    n_features = X_.shape[1]
    n_hidden = int(hidden_nodes_0)
    n_hidden1 = int(hidden_nodes_1)
    learningrate = float(learnrate)
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, n_hidden1)
    #b2_ = np.zeros(1)
    b2_ = np.zeros(n_hidden1)
    W3_ = np.random.randn(n_hidden1, 1)
    b3_ = np.zeros(1)
    # Neural network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()
    W3, b3 = Input(), Input()
    l1 = Linear(X, W1, b1)
    s1 = Relu(l1)
    l2 = Linear(s1, W2, b2)
    s2 = Relu(l2)
    l3 = Linear(s2, W3, b3)
    cost = MSE(y, l3)
    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_,
        W3: W3_,
        b3: b3_
    }
    epochs = int(epochs)
    # Total number of examples
    m = X_.shape[0]
    batch_size = 128
    steps_per_epoch = m // batch_size
    graph = topological_sort(feed_dict)
    trainables = [W1, b1, W2, b2, W3, b3]
    print("Total number of examples = {}".format(m))
    # Step 4
    for i in range(epochs):
        loss = 0
        V_loss = 0
        for j in range(steps_per_epoch):
            # Step 1
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)
            # Reset value of X and y Inputs
            X.value = X_batch
            y.value = y_batch
            # Step 2
            forward_and_backward(graph)
            # Step 3
            sgd_update(trainables, learningrate)
            loss += graph[-1].value

            #validation loss calculate
            X.value = V_X_
            y.value = V_y_
            forward_and_backward(graph)
            V_loss += graph[-1].value
        print("Epoch: {0}, Train Loss: {1}, Valid Loss: {2}".format(i+1, loss/steps_per_epoch, V_loss/steps_per_epoch))
    print(trainables)
    return trainables


def run(learningrate, epochs, hidden_nodes_0, hidden_nodes_1):
    #read data from dataset and prepara for the input data
    features, targets, test_features, test_targets, scaled_features, test_data, rides = read_data()
    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]
    value = nn(train_features, train_targets, val_features, val_targets , hidden_nodes_0, hidden_nodes_1, learningrate, epochs)

if __name__ == "__main__":
    script, first, second, third, fourth = argv
    print(script, first, second, third, fourth)
    run(first, second, third, fourth)
