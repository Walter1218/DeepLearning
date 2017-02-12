"""
Check out the new network architecture and dataset!

Notice that the weights and biases are
generated randomly.

No need to change anything, but feel free to tweak
to test your network, play around with the epochs, batch size, etc!
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']
# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
n_features = X_.shape[1]
n_hidden = 10
n_hidden1 = 2
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
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
s2 = Sigmoid(l2)
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
epochs = 100
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size
graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2, W3, b3]
print("Total number of examples = {}".format(m))
# Step 4
for i in range(epochs):
    loss = 0
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
        sgd_update(trainables)
        loss += graph[-1].value
    print("Epoch: {0}, Loss: {1}".format(i+1, (np.sum(loss)/steps_per_epoch)))
    #print('%10.3f' % pi)
    #print("epoch:"+ str(i+1)+", Loss: {}" .format (np.sum(loss)/steps_per_epoch))
