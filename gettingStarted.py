'''
This part is from the original document of Getting Started with TensorFlow
'''

# Tensors

'''
Tensors are central units of data in TensorFlow, which consist of
a set of primitive values shaped into an array of any number of
dimensions. A tensor's rank is its number of dimensions.
'''

# Here are some examples of tensors:

3 # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

'''
TensorFlow Core tutorial
'''

import tensorflow as tf # importing TensorFlow

# The Computational Graph

'''
A computational graph is a series of TensorFlow operations arrangedd into a
graph of nodes. 
'''
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# To evaluate the nodes, we must run the computational graph within a session.
sess = tf.Session()
print(sess.run([node1, node2]))

# A more complicated computation by combining Tensor nodes with operations
# (operations are also nodes). 
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# A graph can be parameterized to accept external inputs, known as placeholders, 
# which is basically a promise to provide a value later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variables enable us to add trainable parameters to a graph. 
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b # Define a simple linear model

# Variable initializations
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x, [1, 2, 3, 4]}))

# Compute the classical squared loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# The model performance can be improved by tuning the model parameters
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

"""
tf.train API
"""

# TensorFlow provides optimizers that slowly change each variable in order
# to minimize the loss function, here we use the simple gradient descent 
# optimizer function tf.gradients
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))


