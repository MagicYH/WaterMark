import tensorflow as tf

class Model():
    def __init__(self, modelPath = None, summaryPath = None, inputPath = None):
        self._modelPath = modelPath
        self._summaryPath = summaryPath
        self._inputPath = inputPath
        self._width = 80
        self._height = 80
        self._in_channels = 3

    def Train(self, loop_count):

        if self._inputPath is None:
            raise Exception("Input path can't be empty under train model")
        
        with tf.Session() as self._sess:
            if self._modelPath is None:
                soft_max, train, keep_prob, loss = self.BuildModel()
            else:
                saver = tf.train.import_meta_graph(self._modelPath + ".meta")
                saver.restore(self._sess, self._modelPath)
                graph = tf.get_default_graph()

                # soft_max = graph.get_tensor_by_name('soft_max:0')
                soft_max = tf.get_collection('soft_max')[0]
                train = tf.get_collection('train')[0]
                keep_prob = tf.get_collection('keep_prob')[0]
                loss = tf.get_collection('loss')[0]
            
            self._init_data_reader()
            self._sess.run(tf.global_variables_initializer())

            for i in range(loop_count):
                img, label = self._next_batch();
                self._sess.run([train], feed_dict = {x: img, label: label})
                if i % 20 == 19:
                    current_loss = self._sess.run([loss], feed_dict = {x: img, label: label})
                    print('step %d, training loss %g' % (i, current_loss))

    def BuildModel(self):
        """Build identify water mark model with vgg
        """
        x = tf.placeholder(tf.float32, [None, self._width, self._height, self._in_channels])
        label = tf.placeholder(tf.float32, [None, 2])
        tf.add_to_collection('x', x)
        tf.add_to_collection('label', label)

        conv1_1 = self._conv_layer(x,  self._in_channels, 64, 'conv1_1')
        conv1_2 = self._conv_layer(conv1_1, 64, 64, 'conv1_2')
        pool1 = self._max_pool(conv1_2, 'pool1')

        conv2_1 = self._conv_layer(pool1, 64, 128, 'conv2_1')
        conv2_2 = self._conv_layer(conv2_1, 128, 128, 'conv2_2')
        pool2 = self._max_pool(conv2_2, 'pool2')

        conv3_1 = self._conv_layer(pool2, 128, 256, 'conv3_1')
        conv3_2 = self._conv_layer(conv3_1, 256, 256, 'conv3_2')
        pool3 = self._max_pool(conv3_2, 'pool3')

        conv4_1 = self._conv_layer(pool3, 256, 512, 'conv4_1')
        conv4_2 = self._conv_layer(conv4_1, 512, 512, 'conv4_2')
        pool4 = self._max_pool(conv4_2, 'pool4')

        conv5_1 = self._conv_layer(pool4, 512, 512, 'conv5_1')
        conv5_2 = self._conv_layer(conv5_1, 512, 512, 'conv5_2')
        conv5_3 = self._conv_layer(conv5_2, 512, 512, 'conv5_3')
        pool5 = self._max_pool(conv5_3, 'pool5')

        # keep probility
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        tf.add_to_collection('keep_prob', keep_prob)

        fc1 = self._fc_layer(pool5, 25, 4096, 'fc1')
        fc1_relu = tf.nn.relu(fc1)
        fc1_out = tf.nn.dropout(fc1_relu, keep_prob)

        fc2 = self._fc_layer(fc1_out, 4096, 1024, 'fc2')
        fc2_relu = tf.nn.relu(fc2)
        fc2_out = tf.nn.dropout(fc2_relu, keep_prob)

        fc3 = self._fc_layer(fc2_out, 1024, 2, 'fc3')
        soft_max = tf.nn.softmax(fc3, name = 'soft_max')
        tf.add_to_collection('soft_max', soft_max)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=soft_max, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('loss', loss)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
        tf.add_to_collection('train', train)
        
        # summary data
        with tf.name_scope('summaries'):
            tf.summary.scalar('mean', loss)
            tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(soft_max - loss))))

        return soft_max, train, keep_prob, loss

    def _conv_layer(self, input, in_channel, out_channel, name):
        with tf.name_scope(name):
            w, b = self._get_conv_var(in_channel, out_channel, name)
            conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)
            relu = tf.nn.relu(bias)
            return relu

    def _get_conv_var(self, in_channel, out_channel, name):
        initial = tf.truncated_normal([3, 3, in_channel, out_channel], 0.0, 0.001)
        w = tf.Variable(initial, name=name + "_w")

        initial = tf.truncated_normal([out_channel], 0.0, 0.001)
        b = tf.Variable(initial, name=name + "_b")
        return w, b

    def _max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _fc_layer(self, input, in_size, out_size, name):
        with tf.name_scope(name):
            w, b = self._get_fc_var(in_size, out_size, name)
            x = tf.reshape(input, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
            return fc

    def _get_fc_var(self, in_size, out_size, name):
        initial = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        w = tf.variable(initial, name + "_w")

        initial = tf.truncated_normal([out_size], .0, .001)
        b = tf.variable(initial, name + "_b")
        return w, b

    def _init_data_reader(self):
        queue = tf.train.string_input_producer([self._inputPath])

        reader = tf.TFRecordReader()
        _, serialize = reader.read(queue)
        features = tf.parse_single_example(serialize, features = {
            'label': tf.FixedLenFeature([2], tf.int32),
            'img': tf.FixedLenFeature([], tf.string),
        })

        img = tf.decode_raw(features['img'], tf.uint8)
        img = tf.reshape(img, [self._width, self._height, self._in_channels])
        img = tf.cast(img, tf.float32)

        label = features['label']
        label = tf.cast(label, tf.float32)
        
        self._img, self._label = tf.train.shuffle_batch([img, label], batch_size=100, capacity=2000, min_after_dequeue=500)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = self._sess, coord = coord)

    def _next_batch(self):
        img, label = self._sess.run([self._img, self._label])
        return img, label