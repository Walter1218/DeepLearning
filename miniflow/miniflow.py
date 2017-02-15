"""
This project is parts of Udacity's deep learning nanodegree foundation program
This is Miniflow class.
Author: Walter
Date: 11/02/17
"""
import numpy as np

class Node(object):
    def __init__(self, inbound_nodes = []):
        #Receive from others
        self.inbound_nodes = inbound_nodes
        #Output from this Nodes to others
        self.outbound_nodes = []
        # New property! Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        #for each inbound nodes, add this as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation
            Compute the output value based on `inbound_nodes` and
            store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError

"""
Input Node not have the inbound nodes, so we need not do the same operation as in Node class.
This node didi not calculate anything, its just pass the value to next nodes.
"""
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    #NOTE :
    """
    # Input node is the only node where the value
    # may be passed as an argument to forward().
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    """
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero. The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

"""
The Add nodes is just do an addition operate
"""
class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of it's inbound_nodes.
        """
        sizes = len(self.inbound_nodes)
        value = 0
        for i in range(sizes):
            value += self.inbound_nodes[i].value
        self.value = value

"""
This is Mul operation
"""
class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        sizes = len(self.inbound_nodes)
        value = 1
        for i in range(sizes):
            value *= self.inbound_nodes[i].value
        self.value = value

"""
This is Linear Function works for Miniflow,Represents a node that performs a linear transform.
"""
class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.
        """
        # NOTE: this is numpy version of Linear operation
        inputs = np.array(self.inbound_nodes[0].value)
        weights = np.array(self.inbound_nodes[1].value)
        bias = self.inbound_nodes[2].value
        self.value = np.dot(inputs,weights) + bias
        #self.value = np.dot(self.inbound_nodes[0].value, self.inbound_nodes[1].value) + self.inbound_nodes[2].value

        # NOTE: this is not numpy version of Linear operation
        #inputs = self.inbound_nodes[0].value
        #weights = self.inbound_nodes[1].value
        #bias = self.inbound_nodes[2]
        #self.value = bias.value
        #for x, w in zip(inputs, weights):
        #    self.value += x * w

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)

"""
This is Sigmoid function works for our Miniflow Library
"""
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.
        `x`: A numpy array-like object.
        Return the result of the sigmoid function.
        """
        return 1. / (1 + np.exp(-x))
    
    def _sigmoid2deriv(self, x):
        return x * (1 - x)

    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.
        """
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            """
            Set the gradients property to the gradients with respect to each input.
            NOTE: See the Linear node and MSE node for examples.
            """
            self.gradients[self.inbound_nodes[0]] += self._sigmoid2deriv(self.value) * grad_cost
"""
This is Relu function works for Miniflow
"""
class Relu(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    
    def _relu(self, x):
        """
            This method is separate from `forward` because it
            will be used later with `backward` as well.
            `x`: A numpy array-like object.
            Return the result of the relu function.
            """
        return (x > 0) * x
    
    def _relu2deriv(self, x):
        return (x > 0)
    
    def forward(self):
        """
            Set the value of this node to the result of the
            relu function, `_relu`.
            """
        self.value = self._relu(self.inbound_nodes[0].value)
    
    def backward(self):
        """
            Calculates the gradient using the derivative of
            the relu function.
            """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            """
                Set the gradients property to the gradients with respect to each input.
                NOTE: See the Linear node and MSE node for examples.
                """
            self.gradients[self.inbound_nodes[0]] += self._relu2deriv(self.value) * grad_cost
"""
MSE function works for Miniflow
"""
class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        #self.value = np.mean(np.square(self.diff))
        self.value = np.mean(np.square(self.diff))#np.square(self.diff) / self.m

    def backward(self):
        """
        Calculates the gradient of the cost.
        This is the final node of the network so outbound nodes
        are not a concern.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

#NOTE:Here we use the formula for gradient descent (pseudocode) is:
#   x = x - learning_rate * gradient_of_x
#   x is a parameter used by the neural network (i.e. a single weight or bias).
"""
This algorithm using Kahn's algorithm, the reference pages is:
https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
"""
def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    Returns a list of sorted nodes.
    """
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)
    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

"""
It will run the Network and output the results
"""
def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.
    Arguments:
        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.
    Returns the output Node's value
    """
    for n in sorted_nodes:
        n.forward()
    return output_node.value

"""
This function is works for MSE function
"""
def forward_pass_single(graph):
    """
    Performs a forward pass through a list of sorted Nodes.
    Arguments:
        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.
    Arguments:
        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()
    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()

"""
sgd algorithm function works for Miniflow
"""
def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.
    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # access and assign the value of a trainable with `value` attribute.
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]
