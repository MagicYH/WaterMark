import tensorflow as tf

def GetModel(x):
    """Build identify water mark model

    Args:
        x: an input tensor with size (N_examples, 640 * 480), where 640 * 480 is the size of image

    Returns:
        y, output of model
    """
    # with tf.name_scope('reshape'):
    #     x_image = tf.reshape(x, [-1, 640, 480, 1])

    with tf.name_scope('conv1'):
        w_conv1 = weightVariable([5, 3, 1, 64])
        b_conv1 = biasVariable([64])
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    
    with tf.name_scope('pool1'):
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
        h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    with tf.name_scope('conv2'):
        w_conv2 = weightVariable([5, 3, 64, 64])
        b_conv2 = biasVariable([64])
        h_conv2 = tf.nn.relu(conv2d(h_norm1, w_conv2) + b_conv2)
    
    with tf.name_scope('pool2'):
        h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
        h_pool2 = tf.nn.max_pool(h_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
        

    with tf.name_scope('fc1'):
        reshape = tf.reshape(h_pool2, [100, -1])
        dim = reshape.get_shape()[1].value
        weights = weightVariable([dim, 384])
        biases = biasVariable([384])
        h_fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='fc1')

    with tf.name_scope('fc2'):
        w_fc2 = weightVariable([384, 192])
        b_fc2 = biasVariable([192])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2, name='fc2')

    with tf.name_scope('soft_max'):
        w_sm = weightVariable([192, 2])
        b_sm = biasVariable([2])
        y_conv = tf.add(tf.matmul(h_fc2, w_sm), b_sm, name='soft_max')
    
    return y_conv

def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def weightVariable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def biasVariable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)