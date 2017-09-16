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
