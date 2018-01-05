import numpy as np 
import tensorflow as tf 


# define hyperparameters
n_epochs = 2000
minibatch_size = 50
lr = 1e-4
keep = 0.5

# create placeholders
x = tf.placeholder(tf.float32, shape = [None, img_size_cropped,img_size_cropped,num_channels]) #shape in CNNs is always None x height x width x color channels
y_ = tf.placeholder(tf.float32, shape = [None, 10]) #shape is always None x number of classes

# helper functions
def weight_variable(shape):
    """Initializes weights randomly from a normal distribution
    Params: shape: list of dimensionality of the tensor to be initialized
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Initializes the bias term randomly from a normal distribution.
    Params: shape: list of dimensionality for the bias term.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """Performs a convolution over a given patch x with some filter W.
    Uses a stride of length 1 and SAME padding (padded with zeros at the edges)
    Params:
    x: tensor: the image to be convolved over
    W: the kernel (tensor) with which to convolve.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# strides is a length-4 list that specifies the amount to move for each dimension of our input x. 
# the dimensions correspond to the following (in order): batch_size, length of image, width of image, # of channels in image

def max_pool_2x2(x):
    """Performs a max pooling operation over a 2 x 2 region"""
    # ksize: we only want to take the maximum over 1 example and 1 channel. 
    # the middle elements are 2 x 2 because we want to take maxima over 2 x 2 regions
    
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME') # stride 2 right and 2 down

W_conv1 = weight_variable([5, 5, 1, 32]) # 5 x 5 kernel, across an image with 1 channel to 32 channels
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64]) # 5 x 5 kernel, across an "image" with 32 channels to 64 channels
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024]) 
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([256, 10])
b_fc3 = bias_variable([10])
y_out = tf.matmul(h_fc2_drop, W_fc3) + b_fc3



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_out, labels = y_)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, axis = 1), tf.argmax(y_out, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(n_epochs):
        batch = get_batch() #TODO implement this
        training_inputs = batch[0].reshape([minibatch_size,img_size_cropped,img_size_cropped,num_channels])
        training_labels = batch[1]
        if i % 100 == 0:
            print("epoch: {}".format(i))
            train_acc = accuracy.eval(feed_dict = {x: training_inputs, y_: training_labels, keep_prob : 1.0})
            print("training accuracy: {}".format(train_acc))
        sess.run([train_step], feed_dict = {x: training_inputs, y_: training_labels, keep_prob : keep})
    
    test_images = get_test_imgs() #TODO implement this
    test_inputs = test_images.images.reshape([-1,28,28,1])
    test_labels = test_images.labels   
    test_acc = accuracy.eval(feed_dict = {x: test_inputs, y_: test_labels, keep_prob : 1.0})
    print("test accuracy: {}".format(test_acc))