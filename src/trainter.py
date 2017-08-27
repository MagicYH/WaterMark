import tempfile

import tensorflow as tf
from src.model.Identify import GetModel

def Train(inputPath, width, height):
    # Create the model
    x = tf.placeholder(tf.float32, [None, width, height, 1])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    # Build the graph for the deep net
    y_conv, keep_prob = GetModel(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    img, label = read_data(inputPath, width, height)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=50, capacity=1000, min_after_dequeue=500)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Data and model are ready, begin to train")
        for i in range(1000):
            img_data, img_label = sess.run([img_batch, label_batch])
            if i % 2 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                        x: img_data, y_: img_label, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: img_data, y_: img_label, keep_prob: 0.5})

        # print('test accuracy %g' % accuracy.eval(feed_dict={
        #         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def read_data(inputPath, width, height):
    filename_queue = tf.train.string_input_producer([inputPath])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   # return file name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([2], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [width, height, 1])
    img = tf.cast(img, tf.float32)
    # img = tf.cast(img, tf.float32) * (1. / 255)
    # label = tf.cast(features['label'], tf.int64)
    # label = tf.decode_raw(features['img_raw'], tf.uint8)
    # label = tf.reshape(label, [2, 1])
    # label = tf.cast(label, tf.float32)
    label = features['label']
    # label = tf.reshape(label, [2, 1])
    label = tf.cast(label, tf.float32)

    return img, label