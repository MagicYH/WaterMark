import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from src.data.img.ImageHelper import ImageHelper


class Model():
    def __init__(self, modelPath=None, summaryPath=None, inputPath=None):
        self._modelPath = modelPath
        self._summaryPath = summaryPath
        self._inputPath = inputPath
        self._width = 80
        self._height = 80
        # self._in_channels = 3
        self._in_channels = 1
        self._batch_size = 100;

    def BuildData(self, markPath, sourcePath, outPath):
        markImg = Image.open(markPath)
        [markWidth, markHeight] = markImg.size
        tfWriter = tf.python_io.TFRecordWriter(outPath + ".record")
        dWidth = int(self._width / 4)
        dHeight = int(self._height / 4)
        count = 0
        for imgName in os.listdir(sourcePath):
            imgPath = sourcePath + "/" + imgName
            img = Image.open(imgPath)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            [iWidth, iHeight] = img.size
            iHeight = int(400 * iHeight / iWidth)
            iWidth = 400
            img = img.resize((400, iHeight), Image.BICUBIC)
            wNum = int(round((iWidth - self._width) / dWidth))
            hNum = int(round((iHeight - self._height) / dHeight))
            for x in range(wNum):
                for y in range(hNum):
                    regin = (x * dWidth, y * dHeight, x * dWidth +
                             self._width, y * dHeight + self._height)
                    tmpImg = img.crop(regin)

                    count = count + 1
                    label = [1, 0]
                    if count % 2 == 1:
                        label = [0, 1]
                        tmpImg = self._addWater(tmpImg, markImg)
                    if count % 99 == 1:
                        tmpImg.save(outPath + "/" + str(count) + ".png")

                    # [r, g, b] = tmpImg.split()
                    # d = np.append(np.array(r), np.array(g))
                    # d = np.append(d, np.array(b))
                    # imgRaw = d.tobytes()

                    if self._in_channels == 1:
                        tmpImg = tmpImg.convert("L")
                    imgRaw = tmpImg.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                    }))
                    tfWriter.write(example.SerializeToString())

        print("Create %d images" % count)
        tfWriter.close()

    def Train(self, loop_count):

        if self._inputPath is None:
            raise Exception("Input path can't be empty under train model")

        with tf.Session() as self._sess:
            if self._modelPath is None or os.path.exists(self._modelPath) == False:
                print("Build new model")
                fc3, train, keep_prob, loss, x, label = self.BuildModel()
            else:
                print("Recover model from %s" % self._modelPath)
                saver = tf.train.import_meta_graph(self._modelPath + ".meta")
                saver.restore(self._sess, self._modelPath)
                graph = tf.get_default_graph()

                fc3 = tf.get_collection('fc3')[0]
                train = tf.get_collection('train')[0]
                keep_prob = tf.get_collection('keep_prob')[0]
                loss = tf.get_collection('loss')[0]
                x = tf.get_collection('x')[0]
                label = tf.get_collection('label')[0]

            # soft_max, train, keep_prob, loss, x, label = self.BuildModel()
            tf.summary.image('input', x, 5)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('train', self._sess.graph)

            saver = tf.train.Saver()
            self._init_data_reader()
            self._sess.run(tf.global_variables_initializer())

            for i in range(loop_count):
                img, _y = self._next_batch()

                if i % 50 == 0:
                    current_loss, summary = self._sess.run(
                        [loss, merged], feed_dict={x: img, label: _y, keep_prob: 1.0})
                    saver.save(self._sess, self._modelPath)
                    print('step %d, training loss %g' % (i, current_loss))
                    train_writer.add_summary(summary, i)
                else:
                    _, out = self._sess.run([train, fc3], feed_dict={
                                            x: img, label: _y, keep_prob: 0.5})

                # print('step %d' % i)

    def BuildModel(self):
        """Build identify water mark model with vgg
        """
        x = tf.placeholder(
            tf.float32, [None, self._width, self._height, self._in_channels])
        label = tf.placeholder(tf.float32, [None, 2])
        tf.add_to_collection('x', x)
        tf.add_to_collection('label', label)

        # Laplacian change
        # laplace = tf.constant([[[[-1]], [[-1]], [[-1]]],
        #                        [[[-1]], [[8]], [[-1]]],
        #                        [[[-1]], [[-1]], [[-1]]]], dtype=tf.float32)
        laplace = tf.constant([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=tf.float32)
        laplace = tf.reshape(laplace, [3, 3, 1, 1])
        init = tf.nn.conv2d(x, laplace, [1, 1, 1, 1], padding='SAME')
        combind = tf.concat([x, init], 3)

        conv1_1 = self._conv_layer(combind, 5, 5, 2, 16, 'conv1_1')
        # conv1_2 = self._conv_layer(conv1_1, 64, 64, 'conv1_2')
        pool1 = self._max_pool(conv1_1, 'pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        conv2_1 = self._conv_layer(norm1, 5, 5, 16, 32, 'conv2_1')
        # conv2_2 = self._conv_layer(conv2_1, 64, 64, 'conv2_2')
        pool2 = self._max_pool(conv2_1, 'pool2')
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        
        conv3_1 = self._conv_layer(norm2, 3, 3, 32, 64, 'conv3_1')
        # conv3_2 = self._conv_layer(conv3_1, 128, 128, 'conv3_2')
        pool3 = self._max_pool(conv3_1, 'pool3')
        norm3 = tf.nn.lrn(pool3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        conv4_1 = self._conv_layer(norm3, 3, 3, 64, 128, 'conv4_1')
        # conv4_2 = self._conv_layer(conv4_1, 128, 128, 'conv4_2')
        # conv4_3 = self._conv_layer(conv4_2, 128, 128, 'conv4_3')
        norm4 = tf.nn.lrn(conv4_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        pool4 = self._max_pool(norm4, 'pool4')
       

        # conv5_1 = self._conv_layer(pool4, 3, 3, 128, 256, 'conv5_1')
        # # conv5_2 = self._conv_layer(conv5_1, 512, 512, 'conv5_2')
        # # conv5_3 = self._conv_layer(conv5_2, 512, 512, 'conv5_3')
        # norm5 = tf.nn.lrn(conv5_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        # pool5 = self._max_pool(norm5, 'pool5')

        # keep probility
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        tf.add_to_collection('keep_prob', keep_prob)

        fc1 = self._fc_layer(pool4, 25 * 128, 384, 'fc1')
        fc1_relu = tf.nn.relu(fc1)

        fc2 = self._fc_layer(fc1_relu, 384, 192, 'fc2')
        fc2_relu = tf.nn.relu(fc2)
        fc2_out = tf.nn.dropout(fc2_relu, keep_prob)

        fc3 = self._fc_layer(fc2_out, 192, 2, 'fc3')
        tf.add_to_collection('fc3', fc3)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc3, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        train = tf.train.AdamOptimizer(1e-4).minimize(loss)
        tf.add_to_collection('cross_entropy', cross_entropy)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('train', train)
        
        # summary data
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('stddev', tf.sqrt(tf.reduce_mean(tf.square(fc3 - loss))))
            tf.summary.image('init', init, 5)

        return fc3, train, keep_prob, loss, x, label

    def _conv_layer(self, x, width, height,  in_channel, out_channel, name):
        with tf.name_scope(name):
            w, b = self._get_conv_var(
                width, height, in_channel, out_channel, name)
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)
            relu = tf.nn.relu(bias)
            return relu

    def _get_conv_var(self, width, height, in_channel, out_channel, name):
        initial = tf.truncated_normal(
            [width, height, in_channel, out_channel], 0.0, 0.01)
        w = tf.Variable(initial, name=name + "_w")

        initial = tf.truncated_normal([out_channel], 0.0, 0.01)
        b = tf.Variable(initial, name=name + "_b")
        return w, b

    def _max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _fc_layer(self, x, in_size, out_size, name):
        with tf.name_scope(name):
            w, b = self._get_fc_var(in_size, out_size, name)
            x = tf.reshape(x, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, w), b)
            return fc

    def _get_fc_var(self, in_size, out_size, name):
        initial = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
        w = tf.Variable(initial, name + "_w")

        initial = tf.truncated_normal([out_size], 0.0, 0.01)
        b = tf.Variable(initial, name + "_b")
        return w, b

    def _init_data_reader(self):
        queue = tf.train.string_input_producer([self._inputPath + ".record"])

        reader = tf.TFRecordReader()
        _, serialize = reader.read(queue)

        features = tf.parse_single_example(serialize, features = {
            'label': tf.FixedLenFeature([2], tf.int64),
            'img' : tf.FixedLenFeature([], tf.string),
        })

        img = tf.decode_raw(features['img'], tf.uint8)
        img = tf.reshape(img, [self._width, self._height, self._in_channels])
        img = tf.cast(img, tf.float32)

        label = features['label']
        label = tf.cast(label, tf.float32)
        self._img, self._label = tf.train.shuffle_batch([img, label], batch_size=self._batch_size, capacity=2000, min_after_dequeue=500)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = self._sess, coord = coord)

    def _next_batch(self):
        img, label = self._sess.run([self._img, self._label])
        return img, label

    def _addWater(self, tmpImg, markImg):
        percent = random.randint(90, 110)

        [markWidth, markHeight] = markImg.size
        width = int(markWidth * percent / 100)
        height = int(markHeight * percent / 100)
        
        x1 = random.randint(0, self._width - width - 1)
        y1 = random.randint(0, self._height - height - 1)

        x2 = x1 + width
        y2 = y1 + height

        return ImageHelper.AddWaterWithImg(tmpImg, markImg, x1, y1, x2, y2)
